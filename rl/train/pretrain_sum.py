import sys
from pathlib import Path
# add project root to path for imports (go up 2 levels from rl/train/pretrain_sum_prediction.py)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import os
import random
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from tqdm import tqdm
import wandb

from rl.models.policy import CNNPolicy
from rl.train.sft_config import Config
from rl.train.sft_utils import flat_idx_to_extent
from rl.train.sft_dataset import load_and_process_dataset
from rl.train.sft_negatives import generate_negatives_for_positive
from fruit_box import Sum10Env


def pretrain_sum_prediction(
    config: Config,
    policy: CNNPolicy,
    phase1_data: List[Dict],
    device: torch.device,
    checkpoint_dir: str,
    validation_split: float = 0.1,
):
    """Pre-train sum prediction head to learn 'sum must equal 10' rule.
    
    This phase is completely separate from main training epochs.
    Freezes policy heads and trains only feature extractor + sum_prediction_head.
    Uses Phase-1 examples (both legal and illegal) to learn the sum rule.
    
    Args:
        config: Training configuration
        policy: Policy network to pre-train
        phase1_data: List of Phase-1 training examples
        device: Device to run training on
        checkpoint_dir: Directory to save pre-training checkpoint
        validation_split: Fraction of data to use for validation (default: 0.1)
    """
    print("\n" + "="*60)
    print("SUM PREDICTION PRE-TRAINING PHASE")
    print("="*60)
    print(f"Pre-training for {config.sum_pretrain_epochs} epochs")
    print(f"Checkpoint will be saved to: {checkpoint_dir}")
    
    # Split data into train and validation
    random.shuffle(phase1_data)
    val_size = int(len(phase1_data) * validation_split)
    val_data = phase1_data[:val_size]
    train_data = phase1_data[val_size:]
    print(f"Data split: {len(train_data)} train, {len(val_data)} validation ({validation_split*100:.1f}%)")
    
    # Freeze policy heads: phase0_head, phase1_head, anchor_embedding, value_head
    # Trainable: feature_extractor (conv layers, fc, ln), sum_prediction_head
    for name, param in policy.named_parameters():
        if 'phase0_head' in name or 'phase1_head' in name or 'anchor_embedding' in name or 'value_head' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # Create optimizer only for trainable parameters
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=config.sum_pretrain_lr,
        weight_decay=config.weight_decay
    )
    
    print(f"Frozen: phase0_head, phase1_head, anchor_embedding, value_head")
    print(f"Trainable: feature_extractor, sum_prediction_head")
    print(f"Learning rate: {config.sum_pretrain_lr:.2e}")
    print(f"Weight decay: {config.weight_decay:.2e}")
    print(f"Using {len(train_data)} train examples, {len(val_data)} validation examples")
    
    temp_env = Sum10Env()
    
    # Track best validation MAE for early stopping
    best_val_mae = float('inf')
    best_epoch = 0
    final_val_mae = 0.0
    
    # Pre-training loop
    for epoch in range(config.sum_pretrain_epochs):
        print(f"\nPre-training Epoch {epoch + 1}/{config.sum_pretrain_epochs}")
        
        # Shuffle training data
        shuffled_data = train_data.copy()
        random.shuffle(shuffled_data)
        
        # Generate negatives on-the-fly with 1:1 ratio
        all_examples = []
        for pos_example in shuffled_data:
            all_examples.append(pos_example)
            # Generate 1 negative per positive
            negs, _ = generate_negatives_for_positive(pos_example, negative_example_ratio=1.0)
            all_examples.extend(negs)
        
        random.shuffle(all_examples)
        
        epoch_losses = []
        epoch_maes = []
        
        # Training batches
        for batch_idx, start in enumerate(tqdm(
            range(0, len(all_examples), config.sum_pretrain_batch_size),
            desc="Pre-training"
        )):
            batch_data = all_examples[start:start + config.sum_pretrain_batch_size]
            if not batch_data:
                continue
            
            # Stack observations and masks
            batch_obs = torch.stack([d['obs'] for d in batch_data]).to(device)
            batch_masks = torch.stack([d['mask'] for d in batch_data]).to(device)
            
            # Forward pass
            policy.train()
            logits, value, sum_predictions = policy(batch_obs, batch_masks)
            
            # Compute sum prediction loss only
            batch_losses = []
            batch_maes = []
            grids = (batch_obs[:, 0, :, :] * 9.0).cpu().numpy().astype(np.uint8)
            phases = batch_obs[:, 3, 0, 0].cpu().numpy()
            
            # Track examples for detailed logging (first batch of first epoch only)
            log_examples = (epoch == 0 and batch_idx == 0)
            logged_examples = []
            
            for b in range(batch_obs.size(0)):
                if phases[b] > 0.5:  # Phase-1 only
                    # Extract anchor position
                    anchor_mask = batch_obs[b, 2, :, :].cpu().numpy()
                    anchor_pos = np.argwhere(anchor_mask > 0.5)
                    if len(anchor_pos) == 0:
                        continue
                    r1, c1 = int(anchor_pos[0][0]), int(anchor_pos[0][1])
                    
                    # Get grid and compute actual sums
                    grid = grids[b]
                    temp_env.reset(grid=grid.copy())
                    mask = batch_masks[b].cpu().numpy()
                    valid_indices = np.where(mask)[0]
                    
                    actual_sums = np.zeros(170, dtype=np.float32)
                    for extent_idx in valid_indices:
                        r2, c2 = flat_idx_to_extent(r1, c1, extent_idx)
                        if 0 <= r2 < 10 and 0 <= c2 < 17 and r1 <= r2 and c1 <= c2:
                            actual_sum = temp_env.box_sum(r1, c1, r2, c2)
                            actual_sums[extent_idx] = float(actual_sum)
                    
                    # Compute MSE loss
                    if len(valid_indices) > 0:
                        valid_sum_predictions = sum_predictions[b][valid_indices].detach().cpu().numpy()
                        valid_actual_sums_np = actual_sums[valid_indices]
                        valid_actual_sums = torch.from_numpy(valid_actual_sums_np).to(device)
                        mse_loss = F.mse_loss(sum_predictions[b][valid_indices], valid_actual_sums)
                        batch_losses.append(mse_loss)
                        mae = torch.mean(torch.abs(sum_predictions[b][valid_indices] - valid_actual_sums))
                        batch_maes.append(mae.item())
                        
                        # Log detailed examples (first batch, first epoch, first 3 examples)
                        if log_examples and len(logged_examples) < 3:
                            # Find the extent with the largest prediction error for demonstration
                            errors = np.abs(valid_sum_predictions - valid_actual_sums_np)
                            worst_idx = np.argmax(errors)
                            worst_extent_idx = valid_indices[worst_idx]
                            r2, c2 = flat_idx_to_extent(r1, c1, worst_extent_idx)
                            
                            logged_examples.append({
                                'anchor': (r1, c1),
                                'extent': (r2, c2),
                                'extent_idx': worst_extent_idx,
                                'predicted_sum': float(valid_sum_predictions[worst_idx]),
                                'true_sum': float(valid_actual_sums_np[worst_idx]),
                                'error': float(errors[worst_idx]),
                                'is_legal': float(valid_actual_sums_np[worst_idx]) == 10.0,
                            })
            
            if batch_losses:
                loss = torch.stack(batch_losses).mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip_norm)
                optimizer.step()
                
                epoch_losses.append(loss.item())
                epoch_maes.extend(batch_maes)
        
        # Training metrics
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_mae = np.mean(epoch_maes) if epoch_maes else 0.0
        
        # Validation evaluation
        policy.eval()
        val_losses = []
        val_maes = []
        
        # Generate negatives for validation set
        val_examples = []
        for pos_example in val_data:
            val_examples.append(pos_example)
            negs, _ = generate_negatives_for_positive(pos_example, negative_example_ratio=1.0)
            val_examples.extend(negs)
        
        random.shuffle(val_examples)
        
        with torch.no_grad():
            for start in range(0, len(val_examples), config.sum_pretrain_batch_size):
                val_batch = val_examples[start:start + config.sum_pretrain_batch_size]
                if not val_batch:
                    continue
                
                val_obs = torch.stack([d['obs'] for d in val_batch]).to(device)
                val_masks = torch.stack([d['mask'] for d in val_batch]).to(device)
                
                _, _, val_sum_predictions = policy(val_obs, val_masks)
                
                val_grids = (val_obs[:, 0, :, :] * 9.0).cpu().numpy().astype(np.uint8)
                val_phases = val_obs[:, 3, 0, 0].cpu().numpy()
                
                for b in range(val_obs.size(0)):
                    if val_phases[b] > 0.5:
                        anchor_mask = val_obs[b, 2, :, :].cpu().numpy()
                        anchor_pos = np.argwhere(anchor_mask > 0.5)
                        if len(anchor_pos) == 0:
                            continue
                        r1, c1 = int(anchor_pos[0][0]), int(anchor_pos[0][1])
                        
                        grid = val_grids[b]
                        temp_env.reset(grid=grid.copy())
                        mask = val_masks[b].cpu().numpy()
                        valid_indices = np.where(mask)[0]
                        
                        actual_sums = np.zeros(170, dtype=np.float32)
                        for extent_idx in valid_indices:
                            r2, c2 = flat_idx_to_extent(r1, c1, extent_idx)
                            if 0 <= r2 < 10 and 0 <= c2 < 17 and r1 <= r2 and c1 <= c2:
                                actual_sum = temp_env.box_sum(r1, c1, r2, c2)
                                actual_sums[extent_idx] = float(actual_sum)
                        
                        if len(valid_indices) > 0:
                            valid_sum_pred = val_sum_predictions[b][valid_indices]
                            valid_actual = torch.from_numpy(actual_sums[valid_indices]).to(device)
                            val_mse = F.mse_loss(valid_sum_pred, valid_actual)
                            val_mae = torch.mean(torch.abs(valid_sum_pred - valid_actual))
                            val_losses.append(val_mse.item())
                            val_maes.append(val_mae.item())
        
        val_loss = np.mean(val_losses) if val_losses else 0.0
        val_mae = np.mean(val_maes) if val_maes else 0.0
        final_val_mae = val_mae  # Track final validation MAE
        
        # Logging
        print(f"  Train Loss: {avg_loss:.4f}, Train MAE: {avg_mae:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        
        # Log detailed examples (first epoch only)
        if epoch == 0 and logged_examples:
            print(f"\n  Example predictions (showing worst errors):")
            for i, ex in enumerate(logged_examples[:3]):
                status = "✓ LEGAL" if ex['is_legal'] else "✗ ILLEGAL"
                print(f"    Example {i+1}: Anchor=({ex['anchor'][0]},{ex['anchor'][1]}), "
                      f"Extent=({ex['extent'][0]},{ex['extent'][1]}) [{status}]")
                print(f"      Predicted sum: {ex['predicted_sum']:.2f}, "
                      f"True sum: {ex['true_sum']:.2f}, "
                      f"Error: {ex['error']:.2f}")
        
        # Track best validation MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch + 1
        
        # Log to wandb
        log_dict = {
            "pretrain/epoch": epoch + 1,
            "pretrain/train_loss": avg_loss,
            "pretrain/train_mae": avg_mae,
            "pretrain/val_loss": val_loss,
            "pretrain/val_mae": val_mae,
            "pretrain/best_val_mae": best_val_mae,
            "pretrain/best_epoch": best_epoch,
        }
        wandb.log(log_dict)
        
        # Set back to training mode
        policy.train()
    
    # Unfreeze all parameters for main training
    for param in policy.parameters():
        param.requires_grad = True
    
    # Save pre-training checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/policy_sum_pretrained.pt"
    torch.save(policy.state_dict(), checkpoint_path)
    print(f"\n{'='*60}")
    print("PRE-TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation MAE: {best_val_mae:.4f} (epoch {best_epoch})")
    print(f"Final validation MAE: {final_val_mae:.4f}")
    print(f"Saved pre-training checkpoint to: {checkpoint_path}")
    print(f"\nTo use this checkpoint in main training, run:")
    print(f"  python rl/train/train_sft.py --init_checkpoint {checkpoint_path} [other args]")
    print(f"{'='*60}\n")
    
    # Log checkpoint as wandb artifact
    artifact = wandb.Artifact(
        name="sum-pretrained-checkpoint",
        type="model",
        description=f"Sum prediction pre-trained checkpoint after {config.sum_pretrain_epochs} epochs",
    )
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)
    
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Pre-train sum prediction head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoint")
    parser.add_argument("--dataset_name", type=str, default="djdumpling/fruit-box", help="Use full dataset for diversity (not minimal-area)")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--extra_jsonl", type=str, default=None, help="Optional local JSONL with corrective data")
    args = parser.parse_args()
    
    # Initialize wandb
    os.environ["WANDB_DIR"] = tempfile.gettempdir()
    wandb.init(
        project="fruit-box-sft",
        name=f"sum_pretrain_seed{args.seed}",
        config={
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "seed": args.seed,
            "pretrain_epochs": Config().sum_pretrain_epochs,
            "pretrain_lr": Config().sum_pretrain_lr,
            "pretrain_batch_size": Config().sum_pretrain_batch_size,
        },
        tags=["sft", "pretrain", "sum-prediction"],
    )
    print("Wandb initialized!")
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device} | Seed: {args.seed}")
    
    # Load config
    config = Config(seed=args.seed)
    
    # Load and process dataset (only need Phase-1 data)
    print("Loading and processing dataset...")
    _, phase1_data = load_and_process_dataset(
        args.dataset_name,
        args.dataset_split,
        seed=args.seed,
        include_negative_examples=False,  # We'll generate negatives during pre-training
        negative_example_ratio=0.0,
        extra_jsonl=args.extra_jsonl,
    )
    
    # Create model
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170, dropout=config.dropout).to(device)
    print("Model created from scratch")
    
    # Run pre-training with validation split
    checkpoint_path = pretrain_sum_prediction(
        config,
        policy,
        phase1_data,
        device,
        args.checkpoint_dir,
        validation_split=0.1,  # 10% validation split
    )
    
    wandb.finish()
    print(f"\nPre-training complete! Checkpoint saved to: {checkpoint_path}")

if __name__ == "__main__":
    main()