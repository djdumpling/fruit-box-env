#!/usr/bin/env python3
"""
Lightweight script to finetune SFT policy on diverse_1k dataset.

Loads a checkpoint from artifacts/ and trains on out_data/diverse_1k/trajectories.jsonl
using the grid, action, and legal fields to improve action selection.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import os
import random
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from tqdm import tqdm
import wandb

from rl.models.policy import CNNPolicy
from rl.train.train_sft import (
    anchor_to_flat_idx,
    extent_to_flat_idx,
    flat_idx_to_extent,
    build_observation,
    compute_legal_anchors,
    compute_legal_extents,
    get_grid_hash,
    compute_sft_loss,
)


def load_diverse_dataset(jsonl_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load diverse_1k dataset and convert to Phase-0/Phase-1 examples.
    
    Uses the 'legal' field to determine positive/negative examples.
    """
    print(f"Loading dataset from {jsonl_path}...")
    
    phase0_data = []
    phase1_data = []
    
    # Cache for legal actions
    legal_anchors_cache = {}
    legal_extents_cache = {}
    
    with open(jsonl_path, 'r') as f:
        for line in tqdm(f, desc="Processing examples"):
            row = json.loads(line)
            
            grid = np.array(row['grid'], dtype=np.uint8)
            action = row['action']
            r1 = action['r1']
            c1 = action['c1']
            r2 = action['r2']
            c2 = action['c2']
            is_legal = row.get('legal', True)
            
            # Skip invalid coordinates
            if not (0 <= r1 < 10 and 0 <= c1 < 17 and 0 <= r2 < 10 and 0 <= c2 < 17):
                continue
            if not (r1 <= r2 and c1 <= c2):
                continue
            
            grid_hash = get_grid_hash(grid)
            
            # Phase-0: anchor selection
            if grid_hash not in legal_anchors_cache:
                legal_anchors_cache[grid_hash] = compute_legal_anchors(grid)
            legal_anchors_set = legal_anchors_cache[grid_hash]
            
            anchor_idx = anchor_to_flat_idx(r1, c1)
            if anchor_idx not in legal_anchors_set and is_legal:
                # Skip if positive example has illegal anchor
                continue
            
            phase0_obs = build_observation(grid, phase=0, selected_anchor=None)
            phase0_mask = torch.ones(170, dtype=torch.bool)  # All anchors allowed
            
            phase0_data.append({
                'obs': torch.from_numpy(phase0_obs).float(),
                'action': torch.tensor(anchor_idx, dtype=torch.long),
                'mask': phase0_mask,
                'is_positive': is_legal,
                'grid': grid.copy(),
                'legal_anchors_set': legal_anchors_set.copy(),
                'phase': 0,
            })
            
            # Phase-1: extent selection
            cache_key = (grid_hash, anchor_idx)
            if cache_key not in legal_extents_cache:
                legal_extents_cache[cache_key] = compute_legal_extents(grid, r1, c1)
            legal_extents_set = legal_extents_cache[cache_key]
            
            extent_idx = extent_to_flat_idx(r1, c1, r2, c2)
            # Skip (0,0) extents - single cell can't sum to 10
            # But if it's a negative example, we might want to include it
            # However, extent_idx=0 is not in the valid range for phase1_mask (which starts at idx=1)
            # So we skip it for both positive and negative examples
            if extent_idx == 0:
                continue  # Skip (0,0) extents (single cell can't sum to 10)
            
            if extent_idx not in legal_extents_set and is_legal:
                # Skip if positive example has illegal extent
                continue
            
            max_valid_count = (10 - r1) * (17 - c1)
            phase1_mask = torch.zeros(170, dtype=torch.bool)
            for idx in range(1, min(max_valid_count, 170)):
                phase1_mask[idx] = True
            
            phase1_obs = build_observation(grid, phase=1, selected_anchor=(r1, c1))
            
            phase1_data.append({
                'obs': torch.from_numpy(phase1_obs).float(),
                'action': torch.tensor(extent_idx, dtype=torch.long),
                'mask': phase1_mask,
                'anchor': torch.tensor(anchor_idx, dtype=torch.long),
                'is_positive': is_legal,
                'grid': grid.copy(),
                'r1': r1,
                'c1': c1,
                'legal_extents_set': legal_extents_set.copy(),
                'phase': 1,
            })
    
    print(f"Loaded {len(phase0_data)} Phase-0 examples ({sum(1 for d in phase0_data if d['is_positive'])} positive, {sum(1 for d in phase0_data if not d['is_positive'])} negative)")
    print(f"Loaded {len(phase1_data)} Phase-1 examples ({sum(1 for d in phase1_data if d['is_positive'])} positive, {sum(1 for d in phase1_data if not d['is_positive'])} negative)")
    
    return phase0_data, phase1_data


def train(
    checkpoint_path: str,
    dataset_path: str,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    seed: int = 42,
    checkpoint_interval: int = 5,
):
    """Main training function."""
    # Initialize wandb
    os.environ["WANDB_DIR"] = tempfile.gettempdir()
    wandb.init(
        project="fruit-box-sft",
        name=f"diverse_finetune_seed{seed}",
        config={
            "checkpoint_path": checkpoint_path,
            "dataset_path": dataset_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "checkpoint_interval": checkpoint_interval,
        },
        tags=["sft", "fruit-box", "diverse", "finetune"],
    )
    print("Wandb initialized!")
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {seed}")
    
    # Load dataset
    if not Path(dataset_path).exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        wandb.finish()
        return
    
    phase0_data, phase1_data = load_diverse_dataset(dataset_path)
    
    if len(phase0_data) == 0 and len(phase1_data) == 0:
        print("Error: No data loaded!")
        wandb.finish()
        return
    
    # Create model
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    
    # Load checkpoint
    if Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found, training from scratch")
    
    # Optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-5)
    
    # Combine data
    all_data = phase0_data + phase1_data
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Shuffle data
        random.shuffle(all_data)
        
        policy.train()
        epoch_losses = []
        epoch_accuracies = []
        epoch_negative_accuracies = []
        epoch_positive_counts = []
        epoch_negative_counts = []
        epoch_illegal_masses = []
        epoch_topk_illegals = []
        epoch_legal_masses = []
        epoch_legal_predictions = []
        epoch_total_predictions = []
        
        # Process in batches
        for batch_start in tqdm(range(0, len(all_data), batch_size), desc="Training"):
            batch_data = all_data[batch_start:batch_start + batch_size]
            if not batch_data:
                continue
            
            # Extract legal actions sets
            legal_actions_sets = []
            for d in batch_data:
                if d.get('phase') == 0:
                    legal_actions_sets.append(d.get('legal_anchors_set', set()))
                else:
                    legal_actions_sets.append(d.get('legal_extents_set', set()))
            
            # Stack batches
            try:
                batch_obs = torch.stack([d['obs'] for d in batch_data]).to(device)
                batch_actions = torch.stack([d['action'] for d in batch_data]).to(device)
                batch_masks = torch.stack([d['mask'] for d in batch_data]).to(device)
                batch_is_positive = torch.tensor([d.get('is_positive', True) for d in batch_data], dtype=torch.bool).to(device)
            except Exception as e:
                print(f"Error stacking batch: {e}")
                print(f"Batch size: {len(batch_data)}")
                continue
            
            # Forward pass
            loss, info = compute_sft_loss(
                policy,
                batch_obs,
                batch_actions,
                batch_masks,
                batch_is_positive,
                negative_loss_weight=2.0,
                legal_actions_sets=legal_actions_sets,
                illegal_mass_alpha=2.0,
                illegal_mass_beta=3.0,
                topk_illegal_k=10,
                topk_illegal_delta=5.0,
                legal_mass_bonus_zeta=0.5,
                use_set_based_losses=True,
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(info['loss'])
            epoch_accuracies.append(info['accuracy'])
            epoch_negative_accuracies.append(info.get('negative_accuracy', 0.0))
            epoch_positive_counts.append(info.get('positive_count', 0))
            epoch_negative_counts.append(info.get('negative_count', 0))
            epoch_illegal_masses.append(info.get('illegal_mass', 0.0))
            epoch_topk_illegals.append(info.get('topk_illegal', 0.0))
            epoch_legal_masses.append(info.get('legal_mass', 0.0))
            epoch_legal_predictions.append(info.get('legal_predictions', 0))
            epoch_total_predictions.append(info.get('total_predictions', 0))
        
        # Compute epoch metrics
        avg_loss = np.mean(epoch_losses)
        avg_accuracy = np.mean(epoch_accuracies)
        avg_negative_accuracy = np.mean(epoch_negative_accuracies) if epoch_negative_accuracies else 0.0
        total_positive = sum(epoch_positive_counts)
        total_negative = sum(epoch_negative_counts)
        avg_illegal_mass = np.mean(epoch_illegal_masses)
        avg_topk_illegal = np.mean(epoch_topk_illegals)
        avg_legal_mass = np.mean(epoch_legal_masses)
        total_legal_predictions = sum(epoch_legal_predictions)
        total_predictions = sum(epoch_total_predictions)
        legality_rate = (total_legal_predictions / total_predictions) if total_predictions > 0 else 0.0
        
        print(f"  Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, Legality rate: {legality_rate:.4f}")
        if total_negative > 0:
            print(f"  Positive examples: {total_positive}, Negative examples: {total_negative}")
            print(f"  Negative accuracy: {avg_negative_accuracy:.4f}")
        print(f"  Set-based losses: Illegal mass={avg_illegal_mass:.4f}, Top-K illegal={avg_topk_illegal:.4f}, Legal mass={avg_legal_mass:.4f}")
        
        # Log to wandb
        log_dict = {
            "epoch": epoch + 1,
            "train/loss": avg_loss,
            "train/accuracy": avg_accuracy,
            "train/legality_rate": legality_rate,
            "train/illegal_mass": avg_illegal_mass,
            "train/topk_illegal": avg_topk_illegal,
            "train/legal_mass": avg_legal_mass,
        }
        if total_negative > 0:
            log_dict["train/negative_accuracy"] = avg_negative_accuracy
            log_dict["train/positive_count"] = total_positive
            log_dict["train/negative_count"] = total_negative
        wandb.log(log_dict)
        
        # Save checkpoint as wandb artifact (not locally)
        if (epoch + 1) % checkpoint_interval == 0 or epoch == epochs - 1:
            try:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                    tmp_path = tmp_file.name
                    torch.save(policy.state_dict(), tmp_path)
                    
                    # Create and upload wandb artifact
                    artifact = wandb.Artifact(
                        name=f"diverse-checkpoint-epoch-{epoch+1}",
                        type="model",
                        description=f"Diverse finetune checkpoint at epoch {epoch+1}",
                    )
                    artifact.add_file(tmp_path)
                    wandb.log_artifact(artifact)
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    print(f"  Saved checkpoint as wandb artifact: diverse-checkpoint-epoch-{epoch+1}")
            except Exception as e:
                print(f"  Warning: Failed to save checkpoint: {e}")
    
    # Save final checkpoint
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            tmp_path = tmp_file.name
            torch.save(policy.state_dict(), tmp_path)
            
            artifact = wandb.Artifact(
                name="diverse-checkpoint-final",
                type="model",
                description=f"Final diverse finetune checkpoint after {epochs} epochs",
            )
            artifact.add_file(tmp_path)
            wandb.log_artifact(artifact)
            
            os.unlink(tmp_path)
            print(f"  Saved final checkpoint as wandb artifact: diverse-checkpoint-final")
    except Exception as e:
        print(f"  Warning: Failed to save final checkpoint: {e}")
    
    print("\nTraining complete!")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Finetune SFT policy on diverse_1k dataset")
    parser.add_argument("--checkpoint", type=str, default="artifacts/policy_sft_epoch80.pt",
                        help="Path to initial checkpoint")
    parser.add_argument("--dataset", type=str, default="out_data/diverse_1k/trajectories.jsonl",
                        help="Path to diverse_1k trajectories JSONL")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_interval", type=int, default=20, help="Save checkpoint every N epochs")
    args = parser.parse_args()
    
    train(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()

