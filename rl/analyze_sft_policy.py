import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
from datasets import load_dataset
from typing import List, Tuple, Optional
from collections import Counter

from rl.models.policy import CNNPolicy
from rl.envs.sum10_env import Sum10GymEnv
from rl.envs.split_wrapper import TwoPhaseWrapper
from fruit_box import Sum10Env
from rl.train_sft import build_observation, flat_idx_to_anchor, flat_idx_to_extent


def load_checkpoint_from_wandb(artifact_path: str) -> str:
    """Download checkpoint from wandb artifact and return local path.
    
    Args:
        artifact_path: Wandb artifact path (e.g., 'djdumpling-yale/fruit-box-sft/sft-checkpoint-epoch-40:v5')
    
    Returns:
        Local path to the checkpoint file
    """
    import wandb
    
    print(f"Downloading wandb artifact: {artifact_path}")
    # Initialize wandb run to access artifacts
    run = wandb.init()
    artifact = run.use_artifact(artifact_path, type='model')
    artifact_dir = artifact.download()
    
    # Find the checkpoint file in the artifact directory
    artifact_path_obj = Path(artifact_dir)
    checkpoint_files = list(artifact_path_obj.glob("*.pt")) + list(artifact_path_obj.glob("*.pth"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint file (.pt or .pth) found in artifact directory: {artifact_dir}")
    
    if len(checkpoint_files) > 1:
        print(f"Warning: Multiple checkpoint files found, using: {checkpoint_files[0]}")
    
    checkpoint_path = str(checkpoint_files[0])
    print(f"Downloaded checkpoint to: {checkpoint_path}")
    return checkpoint_path


def print_grid(grid: np.ndarray, title: str = "Grid"):
    """Print grid in a readable format"""
    print(f"\n{title}:")
    print("=" * 50)
    for row in grid:
        print(" ".join(f"{cell:2d}" for cell in row))
    print("=" * 50)


def generate_random_grid(seed: Optional[int] = None) -> np.ndarray:
    # unconstrained that sum is multiple of 10, but shouldn't matter for tests
    rng = np.random.default_rng(seed)
    grid = rng.integers(1, 10, size=(10, 17), dtype=np.uint8)
    return grid


def load_initial_grids_from_dataset(
    dataset_name: str = "djdumpling/fruit-box-minimal-area",
    dataset_split: str = "train",
    num_grids: int = 10,
) -> List[np.ndarray]:
    """Load initial grids from HF dataset"""
    hf_dataset = load_dataset(dataset_name, split=dataset_split)
    print(f"Loaded dataset {dataset_name} (split: {dataset_split})...")
    
    # group by episode_id and get unique initial grids
    episodes = {}
    for row in hf_dataset:
        ep_id = row["episode_id"]
        if ep_id not in episodes:
            episodes[ep_id] = row
    
    # extract initial grids
    grids = []
    for ep_id, row in list(episodes.items())[:num_grids]:
        initial_grid = np.array(row["grid"], dtype=np.uint8)
        grids.append(initial_grid)
    
    print(f"Loaded {len(grids)} initial grids")
    return grids


def analyze_sft_policy(
    checkpoint_path: str,
    dataset_name: str = "djdumpling/fruit-box-minimal-area",
    dataset_split: str = "train",
    num_grids: int = 10,
    num_moves_per_grid: int = 5,
    device: Optional[torch.device] = None,
):
    """Run policy on dataset grids and print stats"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # load policy
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy.eval()
    print("Policy loaded successfully!")
    
    # load initial grids
    grids = load_initial_grids_from_dataset(dataset_name, dataset_split, num_grids)
    
    # statistics
    all_rewards = []
    all_valid = []
    reward_distribution = Counter()
    total_moves = 0
    valid_moves = 0
    
    # analyze each grid
    for grid_idx, initial_grid in enumerate(grids):
        print(f"\n{'='*70}")
        print(f"Grid {grid_idx + 1}/{len(grids)}")
        print(f"{'='*70}")
        
        # create environment
        env = Sum10GymEnv(initial_grid=initial_grid.copy())
        wrapped_env = TwoPhaseWrapper(env, curriculum_legal_only=False, curriculum_updates=0)
        
        # create Sum10Env for validation
        validation_env = Sum10Env()
        validation_env.reset(grid=initial_grid.copy())
        
        print_grid(initial_grid, f"Initial Grid {grid_idx + 1}")
        
        # make moves
        obs, info = wrapped_env.reset()
        current_grid = initial_grid.copy()
        
        for move_num in range(num_moves_per_grid):
            print(f"\n--- Move {move_num + 1} ---")
            
            # phase-0: select anchor
            phase0_obs = obs.unsqueeze(0).to(device)  # [1, 4, 10, 17]
            phase0_mask = wrapped_env.get_action_mask().unsqueeze(0).to(device)  # [1, 170]
            
            with torch.no_grad():
                logits, _ = policy(phase0_obs, phase0_mask)
                # use argmax for deterministic evaluation (or sample for stochastic)
                anchor_idx = logits.argmax(dim=1).item()
            
            r1, c1 = flat_idx_to_anchor(anchor_idx)
            print(f"Phase-0: Selected anchor ({r1}, {c1}) [index {anchor_idx}]")
            
            # step Phase-0
            obs, reward, terminated, truncated, info = wrapped_env.step(anchor_idx)
            
            # phase-1: select extent
            phase1_obs = obs.unsqueeze(0).to(device)  # [1, 4, 10, 17]
            phase1_mask = wrapped_env.get_action_mask()  # [valid_count]
            
            # pad mask to 170
            padded_mask = torch.zeros(170, dtype=torch.bool)
            valid_count = phase1_mask.sum().item()
            padded_mask[:valid_count] = phase1_mask
            phase1_mask_padded = padded_mask.unsqueeze(0).to(device)  # [1, 170]
            
            with torch.no_grad():
                logits, _ = policy(phase1_obs, phase1_mask_padded)
                # extract valid logits
                valid_logits = logits[0][:valid_count]
                extent_idx = valid_logits.argmax().item()
            
            r2, c2 = flat_idx_to_extent(r1, c1, extent_idx)
            print(f"Phase-1: Selected extent ({r2}, {c2}) [index {extent_idx}]")
            print(f"Full move: ({r1}, {c1}) → ({r2}, {c2})")
            
            # validate move using Sum10Env
            step_info = validation_env.step(r1, c1, r2, c2)
            
            is_valid = step_info.valid
            reward_value = step_info.reward if is_valid else 0
            actual_sum = step_info.sum
            
            print(f"Validation:")
            print(f"  Valid: {is_valid}")
            print(f"  Sum: {actual_sum} (expected: 10)")
            print(f"  Reward: {reward_value} cells cleared")
            
            if is_valid:
                print(f"  ✓ Move is VALID - cleared {reward_value} cells")
                current_grid = validation_env.grid.copy()
                print_grid(current_grid, f"Grid after move {move_num + 1}")
            else:
                print(f"  ✗ Move is INVALID - sum={actual_sum}, expected=10")
                break
            
            # step Phase-1 in wrapped env
            obs, reward, terminated, truncated, info = wrapped_env.step(extent_idx)
            
            # collect statistics
            all_rewards.append(reward_value)
            all_valid.append(is_valid)
            reward_distribution[reward_value] += 1
            total_moves += 1
            if is_valid:
                valid_moves += 1
            
            # check if episode ended
            if terminated or truncated:
                print(f"\nEpisode ended: terminated={terminated}, truncated={truncated}")
                break
    
    # print statistics
    print(f"\n{'='*70}")
    print("STATISTICS")
    print(f"{'='*70}")
    print(f"Total moves attempted: {total_moves}")
    print(f"Valid moves: {valid_moves}")
    print(f"Legality rate: {valid_moves / max(total_moves, 1):.2%}")
    print(f"\nReward Statistics:")
    if all_rewards:
        print(f"  Average reward per move: {np.mean(all_rewards):.2f} cells")
        print(f"  Median reward per move: {np.median(all_rewards):.2f} cells")
        print(f"  Min reward: {min(all_rewards)} cells")
        print(f"  Max reward: {max(all_rewards)} cells")
        print(f"  Std reward: {np.std(all_rewards):.2f} cells")
        
        print(f"\nReward Distribution:")
        for reward_val in sorted(reward_distribution.keys()):
            count = reward_distribution[reward_val]
            percentage = count / total_moves * 100
            print(f"  {reward_val} cells: {count} moves ({percentage:.1f}%)")
        
        # analyze rectangle sizes
        print(f"\nRectangle Size Analysis:")
        small_rects = sum(1 for r in all_rewards if r <= 2)
        medium_rects = sum(1 for r in all_rewards if 3 <= r <= 5)
        large_rects = sum(1 for r in all_rewards if r >= 6)
        print(f"  Small (≤2 cells): {small_rects} ({small_rects/len(all_rewards)*100:.1f}%)")
        print(f"  Medium (3-5 cells): {medium_rects} ({medium_rects/len(all_rewards)*100:.1f}%)")
        print(f"  Large (≥6 cells): {large_rects} ({large_rects/len(all_rewards)*100:.1f}%)")
    else:
        print("  No valid moves recorded")


def run_policy_on_random_grid(
    checkpoint_path: str,
    num_moves: int = 50,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
):
    """Run policy on a random grid and print stats"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # load policy
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy.eval()
    print("Policy loaded successfully!")
    
    # generate random grid
    initial_grid = generate_random_grid(seed=seed)
    print(f"\n{'='*70}")
    print("RANDOM GRID GENERATION")
    print(f"{'='*70}")
    print(f"Seed: {seed if seed is not None else 'random'}")
    print_grid(initial_grid, "Initial Random Grid")
    
    # Create environment
    env = Sum10GymEnv(initial_grid=initial_grid.copy())
    wrapped_env = TwoPhaseWrapper(env, curriculum_legal_only=False, curriculum_updates=0)
    
    # Create Sum10Env for validation
    validation_env = Sum10Env()
    validation_env.reset(grid=initial_grid.copy())
    
    # statistics
    all_rewards = []
    all_valid = []
    reward_distribution = Counter()
    total_moves = 0
    valid_moves = 0
    
    # run policy
    obs, info = wrapped_env.reset()
    current_grid = initial_grid.copy()
    
    print(f"\n{'='*70}")
    print(f"PLAYING {num_moves} MOVES")
    print(f"{'='*70}")
    
    for move_num in range(num_moves):
        print(f"\n--- Move {move_num + 1}/{num_moves} ---")
        
        # show current grid state
        if verbose:
            print_grid(validation_env.grid.copy(), f"Grid before move {move_num + 1}")
        
        # phase-0: use legal-only anchors (matches SFT training)
        phase0_obs = obs.unsqueeze(0).to(device)  # [1, 4, 10, 17]
        
        # find all anchors that have at least one legal extent
        legal_anchors_set = set()
        for anchor_r1 in range(10):
            for anchor_c1 in range(17):
                anchor_idx = flat_idx_to_anchor(anchor_r1 * 17 + anchor_c1)
                # check if this anchor has any legal extents
                max_valid_count = (10 - anchor_r1) * (17 - anchor_c1)
                has_legal = False
                for extent_idx in range(max_valid_count):
                    r2_test, c2_test = flat_idx_to_extent(anchor_r1, anchor_c1, extent_idx)
                    if validation_env.box_sum(anchor_r1, anchor_c1, r2_test, c2_test) == 10:
                        reward_test = validation_env.box_nonzero_count(anchor_r1, anchor_c1, r2_test, c2_test)
                        if reward_test > 0:
                            has_legal = True
                            break
                if has_legal:
                    legal_anchors_set.add(anchor_r1 * 17 + anchor_c1)
        
        # build mask: True only at positions corresponding to legal anchors
        phase0_mask = torch.zeros(170, dtype=torch.bool)
        for legal_anchor_idx in sorted(legal_anchors_set):
            phase0_mask[legal_anchor_idx] = True
        phase0_mask = phase0_mask.unsqueeze(0).to(device)  # [1, 170]
        
        if phase0_mask.sum() == 0:
            print("  No legal anchors available!")
            break
        
        with torch.no_grad():
            logits, _ = policy(phase0_obs, phase0_mask)
            # extract logits only at legal anchor positions
            legal_anchor_indices = torch.nonzero(phase0_mask[0], as_tuple=False).squeeze(-1)
            valid_logits = logits[0][legal_anchor_indices]
            anchor_idx_compact = valid_logits.argmax().item()
            anchor_idx = legal_anchor_indices[anchor_idx_compact].item()
            # also get top-3 anchors for debugging
            top3_anchors_compact = valid_logits.topk(min(3, valid_logits.numel()))
            top3_anchors = [(legal_anchor_indices[idx].item(), valid_logits[idx].item()) for idx in top3_anchors_compact.indices]
        
        r1, c1 = flat_idx_to_anchor(anchor_idx)
        print(f"Phase-0: Selected anchor ({r1}, {c1}) [index {anchor_idx}]")
        if verbose:
            print(f"  Top-3 anchor logits: {[(flat_idx_to_anchor(idx), logit) for idx, logit in top3_anchors]}")
        
        # step Phase-0
        obs, reward, terminated, truncated, info = wrapped_env.step(anchor_idx)
        
        # phase-1: use legal-only mask (only extents that sum to 10, matches SFT training)
        phase1_obs = obs.unsqueeze(0).to(device)  # [1, 4, 10, 17]
        phase1_mask_legal = wrapped_env.get_legal_only_mask()  # [valid_count] - only legal extents
        
        # pad mask to 170, preserving sparse structure
        padded_mask = torch.zeros(170, dtype=torch.bool)
        valid_indices = torch.nonzero(phase1_mask_legal, as_tuple=False).squeeze(-1)
        if valid_indices.numel() > 0:
            padded_mask[valid_indices] = True
        phase1_mask_padded = padded_mask.unsqueeze(0).to(device)  # [1, 170]
        
        with torch.no_grad():
            logits, _ = policy(phase1_obs, phase1_mask_padded)
            # extract logits only at legal positions
            if valid_indices.numel() > 0:
                valid_count = valid_indices.numel()
                valid_logits = logits[0][valid_indices]
                extent_idx_compact = valid_logits.argmax().item()
                extent_idx = valid_indices[extent_idx_compact].item()
                # also get top-3 extents for debugging
                top3_extents = valid_logits.topk(min(3, valid_count))
            else:
                # no legal moves available
                print("  No legal moves available!")
                break
        
        r2, c2 = flat_idx_to_extent(r1, c1, extent_idx)
        print(f"Phase-1: Selected extent ({r2}, {c2}) [index {extent_idx}/{valid_count-1}]")
        print(f"Full move: Rectangle from ({r1}, {c1}) to ({r2}, {c2})")
        
        # calculate rectangle size
        rect_width = r2 - r1 + 1
        rect_height = c2 - c1 + 1
        rect_size = rect_width * rect_height
        print(f"Rectangle size: {rect_width}x{rect_height} = {rect_size} cells")
        
        if verbose:
            print(f"  Top-3 extent logits: {[(extent_idx, valid_logits[extent_idx].item()) for extent_idx in top3_extents.indices.tolist()]}")
        
        # Validate move using Sum10Env
        step_info = validation_env.step(r1, c1, r2, c2)
        
        is_valid = step_info.valid
        reward_value = step_info.reward if is_valid else 0
        actual_sum = step_info.sum
        
        print(f"Validation:")
        print(f"  Valid: {is_valid}")
        print(f"  Sum: {actual_sum} (expected: 10)")
        print(f"  Reward: {reward_value} cells cleared")
        
        if is_valid:
            print(f"  ✓ Move is VALID - cleared {reward_value} cells")
            current_grid = validation_env.grid.copy()
            if verbose:
                print_grid(current_grid, f"Grid after move {move_num + 1}")
        else:
            print(f"  ✗ Move is INVALID - sum={actual_sum}, expected=10")
            # check what legal moves exist
            legal_moves = validation_env.enumerate_legal()
            print(f"  Legal moves available: {len(legal_moves)}")
            if len(legal_moves) > 0 and verbose:
                print(f"  Sample legal moves (first 5):")
                for (lr1, lc1, lr2, lc2), lreward in legal_moves[:5]:
                    print(f"    ({lr1},{lc1})→({lr2},{lc2}): sum=10, reward={lreward}")
        
        # step Phase-1 in wrapped env
        obs, reward, terminated, truncated, info = wrapped_env.step(extent_idx)
        
        # Collect statistics
        all_rewards.append(reward_value)
        all_valid.append(is_valid)
        reward_distribution[reward_value] += 1
        total_moves += 1
        if is_valid:
            valid_moves += 1
        
        # check if episode ended
        if terminated or truncated:
            print(f"\nEpisode ended: terminated={terminated}, truncated={truncated}")
            if terminated:
                legal_moves = validation_env.enumerate_legal()
                print(f"  Reason: No legal moves available (checked {len(legal_moves)} possible moves)")
            break
    
    # print final grid
    final_grid = validation_env.grid.copy()
    print(f"\n{'='*70}")
    print("FINAL GRID STATE")
    print(f"{'='*70}")
    print_grid(final_grid, "Final Grid")
    
    # print statistics
    print(f"\n{'='*70}")
    print("STATISTICS")
    print(f"{'='*70}")
    print(f"Total moves attempted: {total_moves}")
    print(f"Valid moves: {valid_moves}")
    print(f"Legality rate: {valid_moves / max(total_moves, 1):.2%}")
    print(f"\nReward Statistics:")
    if all_rewards:
        print(f"  Average reward per move: {np.mean(all_rewards):.2f} cells")
        print(f"  Median reward per move: {np.median(all_rewards):.2f} cells")
        print(f"  Min reward: {min(all_rewards)} cells")
        print(f"  Max reward: {max(all_rewards)} cells")
        print(f"  Std reward: {np.std(all_rewards):.2f} cells")
        
        print(f"\nReward Distribution:")
        for reward_val in sorted(reward_distribution.keys()):
            count = reward_distribution[reward_val]
            percentage = count / total_moves * 100
            print(f"  {reward_val} cells: {count} moves ({percentage:.1f}%)")
        
        # analyze rectangle sizes
        print(f"\nRectangle Size Analysis:")
        small_rects = sum(1 for r in all_rewards if r <= 2)
        medium_rects = sum(1 for r in all_rewards if 3 <= r <= 5)
        large_rects = sum(1 for r in all_rewards if r >= 6)
        print(f"  Small (≤2 cells): {small_rects} ({small_rects/len(all_rewards)*100:.1f}%)")
        print(f"  Medium (3-5 cells): {medium_rects} ({medium_rects/len(all_rewards)*100:.1f}%)")
        print(f"  Large (≥6 cells): {large_rects} ({large_rects/len(all_rewards)*100:.1f}%)")
        
        # total cells cleared
        total_cells_cleared = sum(all_rewards)
        print(f"\nTotal cells cleared: {total_cells_cleared} cells")
        print(f"Average cells cleared per valid move: {total_cells_cleared / max(valid_moves, 1):.2f} cells")
    else:
        print("  No valid moves recorded")


def main():
    parser = argparse.ArgumentParser(description="Analyze SFT policy behavior")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to SFT checkpoint or wandb artifact (e.g., 'rl/checkpoints/policy_sft_epoch50.pt' or 'djdumpling-yale/fruit-box-sft/sft-checkpoint-epoch-40:v5')"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="djdumpling/fruit-box-minimal-area",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--num_grids",
        type=int,
        default=10,
        help="Number of initial grids to analyze"
    )
    parser.add_argument(
        "--num_moves",
        type=int,
        default=5,
        help="Number of moves to make per grid"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, defaults to auto)"
    )
    parser.add_argument(
        "--random_grid",
        action="store_true",
        help="Use a random grid instead of dataset grids"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for grid generation (only used with --random_grid)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information for each move"
    )
    
    args = parser.parse_args()
    
    # handle wandb artifact download if needed
    checkpoint_path = args.checkpoint
    if args.checkpoint.startswith("djdumpling") or ("/" in args.checkpoint and ":" in args.checkpoint):
        # Looks like a wandb artifact path
        checkpoint_path = load_checkpoint_from_wandb(args.checkpoint)
    
    device = None
    if args.device:
        device = torch.device(args.device)
    
    if args.random_grid:
        # run on random grid
        run_policy_on_random_grid(
            checkpoint_path=checkpoint_path,
            num_moves=args.num_moves,
            seed=args.random_seed,
            device=device,
            verbose=args.verbose,
        )
    else:
        # run on dataset grids
        analyze_sft_policy(
            checkpoint_path=checkpoint_path,
            dataset_name=args.dataset,
            dataset_split=args.dataset_split,
            num_grids=args.num_grids,
            num_moves_per_grid=args.num_moves,
            device=device,
        )


if __name__ == "__main__":
    main()

