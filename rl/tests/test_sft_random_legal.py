#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.tests.test_sft import get_grid_hash, load_checkpoint_from_wandb
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


def plant_legal_rectangle(grid: np.ndarray, rng: np.random.Generator) -> bool:
    """Insert at least one rectangle whose values sum to 10.

    Returns True on success, False if planting failed after several attempts.
    """
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
            continue  # need at least 2 cells to make sum=10 with digits 1-9
        values = rng.integers(1, 10, size=area - 1)  # Only digits 1-9, no 0s
        needed = 10 - int(values.sum())
        if 1 <= needed <= 9:  # Ensure no 0s
            # fill rectangle row-major
            idx = 0
            for rr in range(r1, r2 + 1):
                for cc in range(c1, c2 + 1):
                    if idx < len(values):
                        grid[rr, cc] = values[idx]
                        idx += 1
                    else:
                        grid[rr, cc] = needed
            return True
    return False


def generate_random_grid(seed: int) -> np.ndarray:
    """Generate a random grid with digits 1-9 (no 0s) and at least one legal move."""
    rng = np.random.default_rng(seed)
    grid = rng.integers(1, 10, size=(10, 17), dtype=np.uint8)  # Only 1-9, no 0s
    if not plant_legal_rectangle(grid, rng):
        # Fallback: create a simple 2x2 rectangle manually summing to 10
        grid.fill(1)  # Fill with 1s first
        grid[0, 0] = 3
        grid[0, 1] = 2
        grid[1, 0] = 2
        grid[1, 1] = 3  # 3+2+2+3 = 10
    return grid


def test_policy_with_legal_masks(
    checkpoint_path: str,
    grids: List[np.ndarray],
    max_moves_per_grid: int = 60,
    verbose: bool = False,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load policy
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check if checkpoint uses old architecture (policy_head) vs new (phase0_head/phase1_head)
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
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Testing on {len(grids)} grids with max {max_moves_per_grid} moves per grid\n")
    
    # Aggregate statistics
    all_grid_results = []
    total_moves = 0
    total_valid_moves = 0
    total_reward = 0
    
    if not grids:
        raise ValueError("No grids provided for evaluation.")
    
    # Use tqdm for progress bar
    for grid_idx, initial_grid in enumerate(tqdm(grids, desc="Testing grids", unit="grid")):
        # Create environments
        env = Sum10GymEnv(initial_grid=initial_grid.copy())
        wrapped_env = TwoPhaseWrapper(env, curriculum_legal_only=False, curriculum_updates=0)
        validation_env = Sum10Env()
        validation_env.reset(grid=initial_grid.copy())
        
        obs, info = wrapped_env.reset()
        
        # Per-grid statistics
        grid_moves = 0
        grid_valid_moves = 0
        grid_reward = 0
        grid_rewards_per_move = []
        grid_terminated = False
        grid_truncated = False
        executed_moves = []  # Track executed moves (r1, c1, r2, c2) for state replay
        
        for move_num in range(max_moves_per_grid):
            # Phase-0: Get legal anchors only
            phase0_obs = obs.unsqueeze(0).to(device)
            phase0_mask = get_legal_anchor_mask(validation_env)  # Only legal anchors
            
            # Pad to 170 if needed
            if phase0_mask.shape[0] < 170:
                padded = torch.zeros(170, dtype=torch.bool)
                padded[:phase0_mask.shape[0]] = phase0_mask
                phase0_mask = padded
            phase0_mask = phase0_mask.unsqueeze(0).to(device)
            
            if phase0_mask.sum() == 0:
                # No legal anchors available - game over
                grid_terminated = True
                break
            
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
            phase1_mask = get_legal_extent_mask(validation_env, r1, c1)  # Only legal extents
            
            # Pad to 170 if needed
            max_valid_count = (10 - r1) * (17 - c1)
            if phase1_mask.shape[0] < 170:
                padded = torch.zeros(170, dtype=torch.bool)
                padded[:phase1_mask.shape[0]] = phase1_mask
                phase1_mask = padded
            phase1_mask = phase1_mask.unsqueeze(0).to(device)
            
            if phase1_mask.sum() == 0:
                grid_moves += 1
                total_moves += 1
                break
            
            with torch.no_grad():
                logits, _, _ = policy(phase1_obs, phase1_mask)
                valid_indices = torch.nonzero(phase1_mask[0], as_tuple=False).squeeze(-1)
                if valid_indices.numel() == 0:
                    grid_moves += 1
                    total_moves += 1
                    break
                valid_logits = logits[0][valid_indices]
                top_idx = torch.argmax(valid_logits).item()
                extent_idx = valid_indices[top_idx].item()
            
            r2, c2 = flat_idx_to_extent(r1, c1, extent_idx)
            
            step_info = validation_env.step(r1, c1, r2, c2)
            is_valid = step_info.valid
            move_reward = step_info.reward if is_valid else 0
            
            if is_valid:
                grid_moves += 1
                total_moves += 1
                grid_valid_moves += 1
                total_valid_moves += 1
                grid_reward += move_reward
                total_reward += move_reward
                grid_rewards_per_move.append(move_reward)
                executed_moves.append((r1, c1, r2, c2))
                
                obs, reward, terminated, truncated, info = wrapped_env.step(extent_idx)
                
                if terminated:
                    grid_terminated = True
                if truncated:
                    grid_truncated = True
                
                if terminated or truncated:
                    break
            else:
                grid_moves += 1
                total_moves += 1
                if verbose:
                    tqdm.write(f"WARNING: Move ({r1},{c1})->({r2},{c2}) was marked as legal but validation failed!")
                break
        
        grid_result = {
            'grid_idx': grid_idx,
            'num_moves': grid_moves,
            'num_valid_moves': grid_valid_moves,
            'total_reward': grid_reward,
            'terminated': grid_terminated,
            'truncated': grid_truncated,
        }
        all_grid_results.append(grid_result)
        
        if verbose:
            tqdm.write(f"Grid {grid_idx + 1}: moves={grid_moves}, reward={grid_reward}")
    
    total_rewards = [r['total_reward'] for r in all_grid_results]
    num_moves_list = [r['num_moves'] for r in all_grid_results]
    
    terminated_count = sum(1 for r in all_grid_results if r['terminated'])
    truncated_count = sum(1 for r in all_grid_results if r['truncated'])
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"\nOverall Statistics:")
    print(f"  Total grids tested: {len(grids)}")
    print(f"  Total moves: {total_moves}")
    print(f"  Total cells cleared: {total_reward}")
    print(f"  Average cells cleared per grid: {np.mean(total_rewards):.2f}")
    
    print(f"\nTotal Cells Cleared per Grid:")
    print(f"  Mean: {np.mean(total_rewards):.2f}")
    print(f"  Median: {np.median(total_rewards):.2f}")
    print(f"  Min: {np.min(total_rewards):.0f}")
    print(f"  Max: {np.max(total_rewards):.0f}")
    print(f"  Std: {np.std(total_rewards):.2f}")
    
    print(f"\nMoves per Grid:")
    print(f"  Mean: {np.mean(num_moves_list):.2f}")
    print(f"  Median: {np.median(num_moves_list):.2f}")
    print(f"  Min: {np.min(num_moves_list)}")
    print(f"  Max: {np.max(num_moves_list)}")
    print(f"  Std: {np.std(num_moves_list):.2f}")
    
    print(f"{'='*70}\n")
    
    results = {
        'total_moves': total_moves,
        'total_valid_moves': total_valid_moves,
        'total_reward': total_reward,
        'num_grids': len(grids),
        'total_reward_stats': {
            'mean': float(np.mean(total_rewards)),
            'median': float(np.median(total_rewards)),
            'min': float(np.min(total_rewards)),
            'max': float(np.max(total_rewards)),
            'std': float(np.std(total_rewards)),
        },
        'moves_stats': {
            'mean': float(np.mean(num_moves_list)),
            'median': float(np.median(num_moves_list)),
            'min': int(np.min(num_moves_list)),
            'max': int(np.max(num_moves_list)),
            'std': float(np.std(num_moves_list)),
        },
        'episode_completion': {
            'terminated': terminated_count,
            'truncated': truncated_count,
            'incomplete': len(grids) - terminated_count - truncated_count,
        },
        'per_grid_results': all_grid_results,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test SFT policy on random grids with legal masks.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SFT checkpoint or wandb artifact")
    parser.add_argument("--num_grids", type=int, default=100, help="Number of random grids to test (default: 100)")
    parser.add_argument("--max_moves", type=int, default=60, help="Maximum moves per grid")
    parser.add_argument("--seed", type=int, default=12345, help="Base RNG seed for grid generation")
    parser.add_argument("--verbose", action="store_true", help="Print per-grid statistics")
    args = parser.parse_args()

    # Generate unique random grids (digits 1-9 only, no 0s)
    grids = []
    seen_hashes = set()
    seed = args.seed
    while len(grids) < args.num_grids:
        grid = generate_random_grid(seed)
        seed += 1
        sig = get_grid_hash(grid)
        if sig in seen_hashes:
            continue
        seen_hashes.add(sig)
        grids.append(grid)

    print(f"Generated {len(grids)} random grids (seed base {args.seed}, digits 1-9 only, no 0s).")

    # Handle wandb artifact download if needed
    checkpoint_path = args.checkpoint
    if args.checkpoint.startswith("djdumpling") or ("/" in args.checkpoint and ":" in args.checkpoint):
        # Looks like a wandb artifact path
        checkpoint_path = load_checkpoint_from_wandb(args.checkpoint)

    results = test_policy_with_legal_masks(
        checkpoint_path=checkpoint_path,
        grids=grids,
        max_moves_per_grid=args.max_moves,
        verbose=args.verbose,
    )
    
    return results


if __name__ == "__main__":
    main()

