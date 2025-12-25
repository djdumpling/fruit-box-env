#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.tests.test_sft import load_checkpoint_from_wandb
from rl.tests.test_sft import flat_idx_to_anchor, flat_idx_to_extent
from rl.models.policy import CNNPolicy
from rl.envs.sum10_env import Sum10GymEnv
from rl.envs.split_wrapper import TwoPhaseWrapper
from fruit_box import Sum10Env


def anchor_to_flat_idx(r1: int, c1: int) -> int:
    """Convert anchor coordinates to flat index (0-169)."""
    return r1 * 17 + c1


def extent_to_flat_idx(r1: int, c1: int, r2: int, c2: int) -> int:
    """Convert extent coordinates to flat index for a given anchor."""
    width = 17 - c1
    dr = r2 - r1
    dc = c2 - c1
    return dr * width + dc


def get_legal_anchor_mask(env: Sum10Env) -> torch.Tensor:
    """Get a mask of legal anchors (anchors that have at least one legal extent)."""
    legal_moves = env.enumerate_legal()
    legal_anchors = set()
    for (r1, c1, r2, c2), _ in legal_moves:
        anchor_idx = anchor_to_flat_idx(r1, c1)
        legal_anchors.add(anchor_idx)
    
    mask = torch.zeros(170, dtype=torch.bool)
    for anchor_idx in legal_anchors:
        mask[anchor_idx] = True
    return mask


def get_legal_extent_mask(env: Sum10Env, r1: int, c1: int) -> torch.Tensor:
    """Get a mask of legal extents for a given anchor."""
    legal_moves = env.enumerate_legal()
    legal_extents = set()
    for (move_r1, move_c1, move_r2, move_c2), _ in legal_moves:
        if move_r1 == r1 and move_c1 == c1:
            extent_idx = extent_to_flat_idx(r1, c1, move_r2, move_c2)
            legal_extents.add(extent_idx)
    
    max_valid_count = (10 - r1) * (17 - c1)
    mask = torch.zeros(max_valid_count, dtype=torch.bool)
    for extent_idx in legal_extents:
        if extent_idx < max_valid_count:
            mask[extent_idx] = True
    return mask


def print_grid_with_action(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int, move_num: int, reward: int):
    """Print grid with the selected action highlighted."""
    print(f"\n{'='*70}")
    print(f"Move {move_num} - Action: ({r1},{c1}) -> ({r2},{c2}) | Reward: {reward} cells cleared")
    print(f"{'='*70}")
    
    # Print column headers
    print("   " + " ".join([f"{i:2d}" for i in range(17)]))
    
    for r in range(10):
        row_str = f"{r:2d} "
        for c in range(17):
            if r1 <= r <= r2 and c1 <= c <= c2:
                # Highlight the selected rectangle
                value = int(grid[r, c])
                if value == 0:
                    row_str += " . "  # Already cleared
                else:
                    row_str += f"[{value:1d}]"  # Selected cell
            else:
                value = int(grid[r, c])
                if value == 0:
                    row_str += " . "
                else:
                    row_str += f" {value:1d} "
        print(row_str)
    
    # Calculate and show sum
    selected_sum = 0
    selected_cells = 0
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            val = grid[r, c]
            if val > 0:
                selected_sum += val
                selected_cells += 1
    
    print(f"\nSelected rectangle: Sum = {selected_sum}, Cells = {selected_cells}, Reward = {reward}")


def print_grid(grid: np.ndarray, title: str = "Grid"):
    """Print grid in a readable format."""
    print(f"\n{title}:")
    print("   " + " ".join([f"{i:2d}" for i in range(17)]))
    for r in range(10):
        row_str = f"{r:2d} "
        for c in range(17):
            value = int(grid[r, c])
            if value == 0:
                row_str += " . "
            else:
                row_str += f" {value:1d} "
        print(row_str)


def test_single_grid(
    checkpoint_path: str,
    grid: np.ndarray,
    max_moves: int = 85,
):
    """Test policy on a single grid with visualization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load policy
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check if checkpoint uses old architecture
    has_old_format = 'policy_head.weight' in checkpoint
    has_new_format = 'phase0_head.weight' in checkpoint
    
    if has_old_format and not has_new_format:
        raise ValueError(
            f"Checkpoint {checkpoint_path} uses an older architecture (policy_head) that is "
            f"incompatible with the current CNNPolicy (phase0_head/phase1_head). "
            f"Please use a checkpoint from a newer training run that matches the current architecture."
        )
    
    policy.load_state_dict(checkpoint)
    policy.eval()
    print(f"Loaded checkpoint from {checkpoint_path}\n")
    
    # Show initial grid
    print_grid(grid, "Initial Grid")
    
    # Create environments
    env = Sum10GymEnv(initial_grid=grid.copy())
    wrapped_env = TwoPhaseWrapper(env, curriculum_legal_only=False, curriculum_updates=0)
    validation_env = Sum10Env()
    validation_env.reset(grid=grid.copy())
    
    obs, info = wrapped_env.reset()
    
    total_reward = 0
    move_num = 0
    
    for move_num in range(1, max_moves + 1):
        # Phase-0: Get legal anchors only
        phase0_obs = obs.unsqueeze(0).to(device)
        phase0_mask = get_legal_anchor_mask(validation_env)
        
        # Pad to 170 if needed
        if phase0_mask.shape[0] < 170:
            padded = torch.zeros(170, dtype=torch.bool)
            padded[:phase0_mask.shape[0]] = phase0_mask
            phase0_mask = padded
        phase0_mask = phase0_mask.unsqueeze(0).to(device)
        
        if phase0_mask.sum() == 0:
            print(f"\n{'='*70}")
            print("No legal moves available. Game over!")
            print(f"{'='*70}")
            break
        
        # Get top-1 anchor choice from policy
        with torch.no_grad():
            logits, _, _ = policy(phase0_obs, phase0_mask)
            valid_indices = torch.nonzero(phase0_mask[0], as_tuple=False).squeeze(-1)
            if valid_indices.numel() == 0:
                break
            valid_logits = logits[0][valid_indices]
            top_idx = torch.argmax(valid_logits).item()
            anchor_idx = valid_indices[top_idx].item()
        
        r1, c1 = flat_idx_to_anchor(anchor_idx)
        
        # Step Phase-0 in wrapped_env
        obs_after_anchor, reward, terminated, truncated, info = wrapped_env.step(anchor_idx)
        
        # Phase-1: Get legal extents only for this anchor
        phase1_obs = obs_after_anchor.unsqueeze(0).to(device)
        phase1_mask = get_legal_extent_mask(validation_env, r1, c1)
        
        # Pad to 170 if needed
        max_valid_count = (10 - r1) * (17 - c1)
        if phase1_mask.shape[0] < 170:
            padded = torch.zeros(170, dtype=torch.bool)
            padded[:phase1_mask.shape[0]] = phase1_mask
            phase1_mask = padded
        phase1_mask = phase1_mask.unsqueeze(0).to(device)
        
        if phase1_mask.sum() == 0:
            print(f"\n{'='*70}")
            print(f"No legal extents for anchor ({r1},{c1}). Game over!")
            print(f"{'='*70}")
            break
        
        # Get top-1 extent choice from policy
        with torch.no_grad():
            logits, _, _ = policy(phase1_obs, phase1_mask)
            valid_indices = torch.nonzero(phase1_mask[0], as_tuple=False).squeeze(-1)
            if valid_indices.numel() == 0:
                break
            valid_logits = logits[0][valid_indices]
            top_idx = torch.argmax(valid_logits).item()
            extent_idx = valid_indices[top_idx].item()
        
        r2, c2 = flat_idx_to_extent(r1, c1, extent_idx)
        
        # Show grid with action before executing (use current grid state)
        current_grid = validation_env.grid.copy()
        
        # Validate move to get reward info
        step_info = validation_env.step(r1, c1, r2, c2)
        move_reward = step_info.reward if step_info.valid else 0
        
        if step_info.valid:
            # Show grid with selected action highlighted (before move)
            print_grid_with_action(current_grid, r1, c1, r2, c2, move_num, move_reward)
            
            total_reward += move_reward
            
            # Step Phase-1 in wrapped_env (this actually executes the move and updates state)
            obs, reward, terminated, truncated, info = wrapped_env.step(extent_idx)
            
            # validation_env already updated by step() above
            
            if terminated:
                print(f"\n{'='*70}")
                print("Game completed! No more legal moves.")
                print(f"{'='*70}")
                break
            if truncated:
                print(f"\n{'='*70}")
                print(f"Max moves ({max_moves}) reached.")
                print(f"{'='*70}")
                break
        else:
            print(f"\n{'='*70}")
            print(f"ERROR: Move ({r1},{c1})->({r2},{c2}) was marked as legal but validation failed!")
            print(f"{'='*70}")
            break
    
    # Show final grid
    print(f"\n{'='*70}")
    print("Final Grid")
    print(f"{'='*70}")
    print_grid(validation_env.grid, "Final State")
    
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Total moves: {move_num}")
    print(f"Total cells cleared: {total_reward}")
    print(f"Average cells per move: {total_reward / move_num:.2f}" if move_num > 0 else "N/A")
    print(f"{'='*70}\n")


def generate_random_grid(seed: int) -> np.ndarray:
    """Generate a random grid with digits 1-9 (no 0s) and at least one legal move."""
    rng = np.random.default_rng(seed)
    grid = rng.integers(1, 10, size=(10, 17), dtype=np.uint8)
    
    # Try to plant at least one legal rectangle
    for _ in range(200):
        r1 = rng.integers(0, 10)
        c1 = rng.integers(0, 17)
        max_r2 = min(r1 + rng.integers(1, 4), 9)
        max_c2 = min(c1 + rng.integers(1, 4), 16)
        if max_r2 <= r1 or max_c2 <= c1:
            continue
        r2 = rng.integers(r1 + 1, max_r2 + 1)
        c2 = rng.integers(c1 + 1, max_c2 + 1)
        area = (r2 - r1 + 1) * (c2 - c1 + 1)
        if area <= 1:
            continue
        values = rng.integers(1, 10, size=area - 1)
        needed = 10 - int(values.sum())
        if 1 <= needed <= 9:
            idx = 0
            for rr in range(r1, r2 + 1):
                for cc in range(c1, c2 + 1):
                    if idx < len(values):
                        grid[rr, cc] = values[idx]
                        idx += 1
                    else:
                        grid[rr, cc] = needed
            return grid
    
    # Fallback
    grid.fill(1)
    grid[0, 0] = 3
    grid[0, 1] = 2
    grid[1, 0] = 2
    grid[1, 1] = 3
    return grid


def main():
    parser = argparse.ArgumentParser(description="Visualize SFT policy on a single grid.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SFT checkpoint or wandb artifact")
    parser.add_argument("--seed", type=int, default=12345, help="Seed for random grid generation")
    parser.add_argument("--max_moves", type=int, default=85, help="Maximum moves per grid")
    parser.add_argument("--grid_file", type=str, default=None, help="Path to numpy file (.npy) containing grid (optional)")
    args = parser.parse_args()
    
    # Load or generate grid
    if args.grid_file:
        grid = np.load(args.grid_file)
        print(f"Loaded grid from {args.grid_file}")
    else:
        grid = generate_random_grid(args.seed)
        print(f"Generated random grid (seed: {args.seed})")
    
    # Handle wandb artifact download if needed
    checkpoint_path = args.checkpoint
    if args.checkpoint.startswith("djdumpling") or ("/" in args.checkpoint and ":" in args.checkpoint):
        checkpoint_path = load_checkpoint_from_wandb(args.checkpoint)
    
    test_single_grid(
        checkpoint_path=checkpoint_path,
        grid=grid,
        max_moves=args.max_moves,
    )


if __name__ == "__main__":
    main()

