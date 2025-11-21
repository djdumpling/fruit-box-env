#!/usr/bin/env python3
"""
Evaluate the SFT policy on randomly generated Sum-10 grids.

Unlike test_sft.py (which replays dataset grids), this script fabricates new
boards by planting at least one legal rectangle per grid, then reuses the
standard evaluation loop from test_sft.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.tests.test_sft import test_policy_with_all_masks  # noqa: E402
from rl.tests.test_sft import get_grid_hash  # reuse helper for consistency
from rl.tests.test_sft import load_checkpoint_from_wandb  # reuse wandb artifact loader
from rl.tests.test_sft import flat_idx_to_anchor, flat_idx_to_extent  # noqa: E402
from rl.models.policy import CNNPolicy  # noqa: E402
from rl.envs.sum10_env import Sum10GymEnv  # noqa: E402
from rl.envs.split_wrapper import TwoPhaseWrapper  # noqa: E402
from fruit_box import Sum10Env  # noqa: E402


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
            continue  # need at least 2 cells to make sum=10 with digits 0-9
        values = rng.integers(0, 10, size=area - 1)
        needed = 10 - int(values.sum())
        if 0 <= needed <= 9:
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
    """Generate a random grid with at least one legal move."""
    rng = np.random.default_rng(seed)
    grid = rng.integers(1, 10, size=(10, 17), dtype=np.uint8)
    if not plant_legal_rectangle(grid, rng):
        # Fallback: create a simple 2x2 rectangle manually summing to 10
        grid.fill(0)
        grid[0, 0] = 5
        grid[0, 1] = 2
        grid[1, 0] = 1
        grid[1, 1] = 2  # 5+2+1+2 = 10
    return grid


def print_grid(grid: np.ndarray, title: str = "Grid"):
    """Print grid in a readable format."""
    print(f"\n{title}:")
    print("   " + " ".join([f"{i:2d}" for i in range(17)]))
    for r in range(10):
        print(f"{r:2d} " + " ".join([f"{int(grid[r, c]):2d}" for c in range(17)]))


def print_action(r1: int, c1: int, r2: int, c2: int, valid: bool, sum_value: int = None, reward: int = None):
    """Print action in a readable format."""
    status = "✓ VALID" if valid else "✗ INVALID"
    area = (r2 - r1 + 1) * (c2 - c1 + 1)
    info = f"Action: ({r1},{c1}) -> ({r2},{c2}) [Area: {area} cells] [{status}]"
    if sum_value is not None:
        info += f" | Sum: {sum_value}"
    if reward is not None and valid:
        info += f" | Reward: {reward}"
    print(info)
    
    # Highlight the selected rectangle
    if not valid:
        print(f"  ⚠️  Selected rectangle is {area} cell(s) - needs at least 2 cells to sum to 10!")


def print_legal_moves(env: Sum10Env, limit: int = 5):
    """Print available legal moves."""
    legal_moves = env.enumerate_legal()
    if not legal_moves:
        print("  No legal moves available.")
        return
    
    print(f"  Available legal moves (showing first {min(limit, len(legal_moves))}):")
    for i, ((r1, c1, r2, c2), reward) in enumerate(legal_moves[:limit]):
        area = (r2 - r1 + 1) * (c2 - c1 + 1)
        print(f"    {i+1}. ({r1},{c1}) -> ({r2},{c2}) [Area: {area}, Reward: {reward}]")
    if len(legal_moves) > limit:
        print(f"    ... and {len(legal_moves) - limit} more")


def test_policy_with_visualization(
    checkpoint_path: str,
    grids: list,
    max_moves_per_grid: int = 60,
):
    """Test policy with detailed visualization of grids and actions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load policy
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint)
    policy.eval()
    print(f"Loaded checkpoint from {checkpoint_path}\n")
    
    for grid_idx, initial_grid in enumerate(grids):
        print("=" * 80)
        print(f"GRID {grid_idx + 1}")
        print("=" * 80)
        
        # Create environments
        env = Sum10GymEnv(initial_grid=initial_grid.copy())
        wrapped_env = TwoPhaseWrapper(env, curriculum_legal_only=False, curriculum_updates=0)
        validation_env = Sum10Env()
        validation_env.reset(grid=initial_grid.copy())
        
        print_grid(initial_grid, "Initial Grid")
        
        # Show available legal moves
        print_legal_moves(validation_env)
        
        obs, info = wrapped_env.reset()
        move_num = 0
        total_valid = 0
        total_reward = 0
        
        for move_num in range(max_moves_per_grid):
            print(f"\n--- Move {move_num + 1} ---")
            
            # Phase-0: Get anchor
            phase0_obs = obs.unsqueeze(0).to(device)
            phase0_mask = wrapped_env.get_action_mask()
            
            if phase0_mask.shape[0] < 170:
                padded = torch.zeros(170, dtype=torch.bool)
                padded[:phase0_mask.shape[0]] = phase0_mask
                phase0_mask = padded
            phase0_mask = phase0_mask.unsqueeze(0).to(device)
            
            if phase0_mask.sum() == 0:
                print("No valid anchors available. Game over.")
                break
            
            # Get top anchor from policy
            with torch.no_grad():
                logits, _, _ = policy(phase0_obs, phase0_mask)  # ignore value and sum_predictions
                valid_indices = torch.nonzero(phase0_mask[0], as_tuple=False).squeeze(-1)
                if valid_indices.numel() == 0:
                    break
                valid_logits = logits[0][valid_indices]
                top_idx_compact = torch.argmax(valid_logits).item()
                anchor_idx = valid_indices[top_idx_compact].item()
            
            r1, c1 = flat_idx_to_anchor(anchor_idx)
            print(f"Phase-0: Selected anchor ({r1}, {c1})")
            
            # Step Phase-0
            obs_after_anchor, _, _, _, _ = wrapped_env.step(anchor_idx)
            
            # Phase-1: Get extent
            phase1_obs = obs_after_anchor.unsqueeze(0).to(device)
            phase1_mask = wrapped_env.get_action_mask()
            
            if phase1_mask.shape[0] < 170:
                padded = torch.zeros(170, dtype=torch.bool)
                padded[:phase1_mask.shape[0]] = phase1_mask
                phase1_mask = padded
            phase1_mask = phase1_mask.unsqueeze(0).to(device)
            
            if phase1_mask.sum() == 0:
                print("No valid extents available. Trying next anchor...")
                continue
            
            # Get top extent from policy
            with torch.no_grad():
                logits, _, _ = policy(phase1_obs, phase1_mask)  # ignore value and sum_predictions
                valid_indices = torch.nonzero(phase1_mask[0], as_tuple=False).squeeze(-1)
                if valid_indices.numel() == 0:
                    continue
                valid_logits = logits[0][valid_indices]
                top_idx_compact = torch.argmax(valid_logits).item()
                extent_idx = valid_indices[top_idx_compact].item()
            
            width = 17 - c1
            r2, c2 = flat_idx_to_extent(r1, c1, extent_idx)
            print(f"Phase-1: Selected extent index {extent_idx} -> ({r2}, {c2})")
            
            # Show top-3 extent choices for debugging
            if valid_indices.numel() > 1:
                k = min(3, valid_indices.numel())
                topk_values, topk_indices_compact = torch.topk(valid_logits, k)
                print(f"  Top-{k} extent choices:")
                for i, (val, idx_compact) in enumerate(zip(topk_values, topk_indices_compact)):
                    ext_idx = valid_indices[idx_compact].item()
                    er2, ec2 = flat_idx_to_extent(r1, c1, ext_idx)
                    area = (er2 - r1 + 1) * (ec2 - c1 + 1)
                    print(f"    {i+1}. Index {ext_idx} -> ({er2}, {ec2}) [Area: {area}, Logit: {val.item():.2f}]")
            
            # Check if action is valid
            step_info = validation_env.step(r1, c1, r2, c2)
            
            print_action(r1, c1, r2, c2, step_info.valid, step_info.sum, step_info.reward)
            
            if not step_info.valid:
                # Show what legal moves were available
                print_legal_moves(validation_env)
            
            if step_info.valid:
                total_valid += 1
                total_reward += step_info.reward
                print_grid(validation_env.grid, f"Grid after move {move_num + 1}")
                obs = torch.from_numpy(validation_env.grid.copy()).float().unsqueeze(0)
                # Update wrapped_env to match
                env = Sum10GymEnv(initial_grid=validation_env.grid.copy())
                wrapped_env = TwoPhaseWrapper(env, curriculum_legal_only=False, curriculum_updates=0)
                wrapped_env.reset()
                obs, _ = wrapped_env.reset()
                
                if step_info.done:
                    print("\nGame completed (no more legal moves)!")
                    break
            else:
                print(f"\nInvalid move! Game ends. Sum was {step_info.sum}, expected 10.")
                break
        
        print(f"\n--- Summary for Grid {grid_idx + 1} ---")
        print(f"Total moves: {move_num + 1}")
        print(f"Valid moves: {total_valid}")
        print(f"Legality rate: {100 * total_valid / (move_num + 1):.1f}%")
        print(f"Total reward: {total_reward}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Test SFT policy on random grids.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SFT checkpoint or wandb artifact")
    parser.add_argument("--num_grids", type=int, default=5, help="Number of random grids to test (default: 5)")
    parser.add_argument("--max_moves", type=int, default=60, help="Maximum moves per grid")
    parser.add_argument("--seed", type=int, default=12345, help="Base RNG seed for grid generation")
    parser.add_argument("--collect_examples", action="store_true", help="Emit corrective examples")
    parser.add_argument("--output_examples", type=str, default=None, help="Path for corrective JSONL")
    parser.add_argument("--verbose", action="store_true", help="Print per-grid stats")
    parser.add_argument("--visualize", action="store_true", help="Show grid and action visualization (default: False, shows summary stats)")
    args = parser.parse_args()

    # Generate unique random grids
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

    print(f"Generated {len(grids)} random grids (seed base {args.seed}).")

    # Handle wandb artifact download if needed
    checkpoint_path = args.checkpoint
    if args.checkpoint.startswith("djdumpling") or ("/" in args.checkpoint and ":" in args.checkpoint):
        # Looks like a wandb artifact path
        checkpoint_path = load_checkpoint_from_wandb(args.checkpoint)

    if args.visualize:
        test_policy_with_visualization(
            checkpoint_path=checkpoint_path,
            grids=grids,
            max_moves_per_grid=args.max_moves,
        )
    else:
        test_policy_with_all_masks(
            checkpoint_path=checkpoint_path,
            grids=grids,
            max_moves_per_grid=args.max_moves,
            verbose=args.verbose,
            collect_examples=args.collect_examples,
            output_examples_path=args.output_examples,
        )


if __name__ == "__main__":
    main()

