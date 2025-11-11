"""
Supervised Fine-Tuning (SFT) for Fruit Box environment.

This script loads trajectories from the Hugging Face dataset and trains a CNNPolicy
model using supervised learning. The trained model can then be used as a starting
point for GRPO training in train_grpo.py.

The model architecture matches train_grpo.py exactly, so checkpoints are compatible.
To load an SFT checkpoint in train_grpo.py, modify the code to load the checkpoint
before training starts:
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    policy.load_state_dict(torch.load("checkpoints/policy_sft_final.pt", map_location=device))

Usage:
    python rl/train_sft.py --seed 42 --epochs 10 --batch_size 64 --lr 1e-4
"""

import sys
from pathlib import Path
# add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
import random
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from datasets import load_dataset
import wandb

from rl.models.policy import CNNPolicy
from fruit_box import Sum10Env


@dataclass
class Config:
    """SFT training configuration."""
    # Data
    dataset_name: str = "djdumpling/fruit-box-minimal-area"
    dataset_split: str = "train"
    
    # Training
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-5
    
    # Other
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 2


def anchor_to_flat_idx(r1: int, c1: int) -> int:
    """Convert anchor (r1, c1) to flat index [0, 169]."""
    return r1 * 17 + c1


def flat_idx_to_anchor(idx: int) -> Tuple[int, int]:
    """Convert flat index [0, 169] to anchor (r1, c1)."""
    r1 = idx // 17
    c1 = idx % 17
    return (r1, c1)


def extent_to_flat_idx(r1: int, c1: int, r2: int, c2: int) -> int:
    """Convert extent (r2, c2) to flat index given anchor (r1, c1).
    
    Valid extents: r2 in [r1, 9], c2 in [c1, 16]
    Flat index: (r2 - r1) * (17 - c1) + (c2 - c1)
    """
    if not (r1 <= r2 < 10 and c1 <= c2 < 17):
        raise ValueError(f"Invalid extent: anchor=({r1},{c1}), extent=({r2},{c2})")
    dr = r2 - r1
    dc = c2 - c1
    width = 17 - c1
    return dr * width + dc


def flat_idx_to_extent(r1: int, c1: int, idx: int) -> Tuple[int, int]:
    """Convert flat index to extent (r2, c2) given anchor (r1, c1)."""
    width = 17 - c1
    dr = idx // width
    dc = idx % width
    r2 = r1 + dr
    c2 = c1 + dc
    return (r2, c2)


def build_observation(grid: np.ndarray, phase: int, selected_anchor: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Build 4-channel observation tensor from grid state.
    
    Args:
        grid: [10, 17] grid array
        phase: 0 for Phase-0 (select anchor), 1 for Phase-1 (select extent)
        selected_anchor: (r1, c1) tuple for Phase-1, None for Phase-0
    
    Returns:
        obs: [4, 10, 17] observation tensor
    """
    grid = grid.astype(np.float32)
    
    # channel 0: normalized values
    value_norm = grid / 9.0
    
    # channel 1: nonzero mask
    nonzero_mask = (grid > 0).astype(np.float32)
    
    # channel 2: anchor mask (zeros in Phase-0, selected anchor=1 in Phase-1)
    anchor_mask = np.zeros((10, 17), dtype=np.float32)
    if phase == 1 and selected_anchor is not None:
        r1, c1 = selected_anchor
        anchor_mask[r1, c1] = 1.0
    
    # channel 3: phase mask (all zeros in Phase-0, all ones in Phase-1)
    phase_mask = np.full((10, 17), float(phase), dtype=np.float32)
    
    obs = np.stack([value_norm, nonzero_mask, anchor_mask, phase_mask], axis=0)
    return obs


def load_and_process_dataset(
    dataset_name: str,
    dataset_split: str,
    seed: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Load dataset and process into Phase-0 and Phase-1 training examples.
    
    Returns:
        phase0_data: List of dicts with keys: 'obs', 'action', 'mask'
        phase1_data: List of dicts with keys: 'obs', 'action', 'mask', 'anchor'
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    hf_dataset = load_dataset(dataset_name, split=dataset_split)
    print(f"Loaded dataset {dataset_name} (split: {dataset_split})...")
    
    # group trajectories by episode_id and agent_tag
    episodes = {}
    for row in hf_dataset:
        ep_id = row["episode_id"]
        agent_tag = row.get("agent_tag", "unknown")
        key = f"{ep_id}_{agent_tag}"
        if key not in episodes:
            episodes[key] = []
        episodes[key].append(row)
    
    for key in episodes:
        episodes[key].sort(key=lambda x: x["step"])
    
    phase0_data = []
    phase1_data = []
    
    # process each trajectory
    for key, trajectory in episodes.items():
        if not trajectory:
            continue
        
        # get initial grid
        initial_state = trajectory[0]
        initial_grid = np.array(initial_state["grid"], dtype=np.uint8)
        
        # simulate environment to build observations
        env = Sum10Env()
        env.reset(grid=initial_grid)
        current_grid = env.grid.copy()
        
        # process each step in the trajectory
        for step in trajectory:
            action = step.get("action", {})
            r1 = action.get("r1", -1)
            c1 = action.get("c1", -1)
            r2 = action.get("r2", -1)
            c2 = action.get("c2", -1)
            
            # skip invalid actions
            if r1 == -1 or c1 == -1 or r2 == -1 or c2 == -1:
                continue
            
            # Phase-0: select anchor (r1, c1)
            phase0_obs = build_observation(current_grid, phase=0, selected_anchor=None)
            phase0_action = anchor_to_flat_idx(r1, c1)
            phase0_mask = torch.ones(170, dtype=torch.bool)  # all anchors valid
            
            phase0_data.append({
                'obs': torch.from_numpy(phase0_obs).float(),
                'action': torch.tensor(phase0_action, dtype=torch.long),
                'mask': phase0_mask,
            })
            
            # Phase-1: select extent (r2, c2) given anchor (r1, c1)
            phase1_obs = build_observation(current_grid, phase=1, selected_anchor=(r1, c1))
            phase1_action_compact = extent_to_flat_idx(r1, c1, r2, c2)
            
            # build action mask for Phase-1
            # pad to max size (170) with True at first valid_count positions
            # this matches the format used in train_grpo.py
            valid_count = (10 - r1) * (17 - c1)
            phase1_mask = torch.zeros(170, dtype=torch.bool)
            phase1_mask[:valid_count] = True
            
            phase1_data.append({
                'obs': torch.from_numpy(phase1_obs).float(),
                'action': torch.tensor(phase1_action_compact, dtype=torch.long),
                'mask': phase1_mask,
                'anchor': torch.tensor(phase0_action, dtype=torch.long),
            })
            
            # execute action to update grid state
            step_info = env.step(r1, c1, r2, c2)
            if step_info.valid:
                current_grid = env.grid.copy()
            else:
                # invalid move - stop processing this trajectory
                break
    
    print(f"Processed {len(phase0_data)} Phase-0 examples and {len(phase1_data)} Phase-1 examples")
    return phase0_data, phase1_data


def compute_sft_loss(
    policy: nn.Module,
    obs: torch.Tensor,
    actions: torch.Tensor,
    masks: torch.Tensor,
) -> Tuple[torch.Tensor, Dict]:
    """Compute supervised fine-tuning loss.
    
    Args:
        policy: CNNPolicy model
        obs: [batch_size, 4, 10, 17] observations
        actions: [batch_size] action indices (compact indices into valid action space)
        masks: [batch_size, 170] action masks (True at first valid_count positions)
    
    Returns:
        loss: Scalar loss tensor
        info: Dictionary with loss components
    """
    logits, _ = policy(obs, masks)  # [batch_size, 170]
    
    # compute loss for each sample
    # masks are padded to 170, with True at first valid_count positions
    losses = []
    correct = 0
    total = 0
    
    for b in range(obs.size(0)):
        mask = masks[b]  # [170]
        valid_count = mask.sum().item()
        
        if valid_count == 0:
            continue
        
        # extract valid logits (first valid_count positions)
        valid_logits = logits[b][:valid_count]  # [valid_action_count]
        action = actions[b].item()
        
        # ensure action is within valid range
        if action >= valid_count:
            # skip invalid actions (shouldn't happen, but handle gracefully)
            continue
        
        # create cross-entropy loss
        loss = F.cross_entropy(valid_logits.unsqueeze(0), torch.tensor([action], device=obs.device))
        losses.append(loss)
        
        # compute accuracy
        pred_action = valid_logits.argmax().item()
        if pred_action == action:
            correct += 1
        total += 1
    
    if len(losses) == 0:
        # return zero loss if no valid samples
        loss = torch.tensor(0.0, device=obs.device, requires_grad=True)
        accuracy = 0.0
    else:
        loss = torch.stack(losses).mean()
        accuracy = correct / total if total > 0 else 0.0
    
    info = {
        'loss': loss.item(),
        'accuracy': accuracy,
    }
    
    return loss, info


def train(config: Config):
    """Main training loop."""
    # initialize wandb (always enabled)
    os.environ["WANDB_DIR"] = tempfile.gettempdir()
    wandb.init(
        project="fruit-box-sft",
        name=f"sft_seed{config.seed}",
        config={
            "dataset_name": config.dataset_name,
            "dataset_split": config.dataset_split,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
        },
        tags=["sft", "fruit-box", "supervised"],
    )
    print("Wandb initialized!")
    
    # set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {config.seed}")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # load and process dataset
    print("Loading and processing dataset...")
    phase0_data, phase1_data = load_and_process_dataset(
        config.dataset_name,
        config.dataset_split,
        seed=config.seed,
    )
    
    # create model
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    print("Model created")
    
    # create optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # training loop
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        
        # combine Phase-0 and Phase-1 data
        all_data = phase0_data + phase1_data
        random.shuffle(all_data)
        
        policy.train()
        epoch_losses = []
        epoch_accuracies = []
        
        for start in tqdm(range(0, len(all_data), config.batch_size), desc="Training"):
            batch_data = all_data[start:start + config.batch_size]
            
            # stack batches (masks already padded to 170)
            batch_obs = torch.stack([d['obs'] for d in batch_data]).to(device)
            batch_actions = torch.stack([d['action'] for d in batch_data]).to(device)
            batch_masks = torch.stack([
                d['mask'] if d['mask'].shape[0] == 170 
                else torch.cat([d['mask'], torch.zeros(170 - d['mask'].shape[0], dtype=torch.bool)])
                for d in batch_data
            ]).to(device)
            
            # forward pass
            loss, info = compute_sft_loss(policy, batch_obs, batch_actions, batch_masks)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(info['loss'])
            epoch_accuracies.append(info['accuracy'])
        
        # logging
        avg_loss = np.mean(epoch_losses)
        avg_accuracy = np.mean(epoch_accuracies)
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_loss,
            "train/accuracy": avg_accuracy,
        })
        
        # checkpoint
        if (epoch + 1) % config.checkpoint_interval == 0:
            checkpoint_path = f"{config.checkpoint_dir}/policy_sft_epoch{epoch+1}.pt"
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            artifact = wandb.Artifact(
                name=f"sft-checkpoint-epoch-{epoch+1}",
                type="model",
                description=f"SFT checkpoint at epoch {epoch+1}",
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
    
    # save final checkpoint
    final_checkpoint_path = f"{config.checkpoint_dir}/policy_sft_final.pt"
    torch.save(policy.state_dict(), final_checkpoint_path)
    print(f"\nTraining complete! Final checkpoint: {final_checkpoint_path}")
    artifact = wandb.Artifact(
        name="sft-checkpoint-final",
        type="model",
        description=f"Final SFT checkpoint after {config.epochs} epochs",
    )
    artifact.add_file(final_checkpoint_path)
    wandb.log_artifact(artifact)
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dataset_name", type=str, default="djdumpling/fruit-box-minimal-area")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=2)
    args = parser.parse_args()
    
    config = Config(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
    )
    
    train(config)


if __name__ == "__main__":
    main()

