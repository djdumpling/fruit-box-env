"""Test SFT policy with all geometrically valid masks (not just legal-only) to see if it learned to avoid illegal actions"""

import sys
from pathlib import Path
# Add project root to path (go up 2 levels from rl/tests/test_sft.py)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import hashlib
from typing import List, Optional, Dict, Set, Tuple
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
from rl.models.policy import CNNPolicy
from rl.envs.sum10_env import Sum10GymEnv
from rl.envs.split_wrapper import TwoPhaseWrapper
from fruit_box import Sum10Env, load_environment


def flat_idx_to_anchor(idx: int):
    r1 = idx // 17
    c1 = idx % 17
    return (r1, c1)


def flat_idx_to_extent(r1: int, c1: int, idx: int):
    width = 17 - c1
    dr = idx // width
    dc = idx % width
    r2 = r1 + dr
    c2 = c1 + dc
    return (r2, c2)


def get_grid_hash(grid: np.ndarray) -> str:
    """Generate a hash for a grid state"""
    return hashlib.md5(grid.tobytes()).hexdigest()


def get_example_key(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> str:
    """Generate a unique key for an example to avoid duplicates"""
    grid_hash = get_grid_hash(grid)
    return f"{grid_hash}_{r1}_{c1}_{r2}_{c2}"


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


def load_grids_from_loader(
    dataset_name: str = "djdumpling/fruit-box-minimal-area",
    dataset_split: str = "train",
    num_grids: int = 10,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Load initial grids using fruit_box.load_environment"""
    env = load_environment(dataset_name=dataset_name, dataset_split=dataset_split, seed=seed)
    dataset = env.dataset
    
    grids: List[np.ndarray] = []
    seen_episodes = set()
    for row in dataset:
        info = row.get("info", {})
        episode_id = info.get("episode_id")
        if episode_id in seen_episodes:
            continue
        seen_episodes.add(episode_id)
        
        initial_grid = info.get("initial_grid")
        if initial_grid is None:
            continue
        grids.append(np.array(initial_grid, dtype=np.uint8))
        if len(grids) >= num_grids:
            break
    
    if not grids:
        raise RuntimeError("No grids loaded from dataset via load_environment()")
    
    if len(grids) < num_grids:
        print(f"Warning: requested {num_grids} grids but only loaded {len(grids)} unique episodes")
    
    return grids


def test_policy_with_all_masks(
    checkpoint_path: str,
    grids: List[np.ndarray],
    max_moves_per_grid: int = 60,
    verbose: bool = False,
    collect_examples: bool = False,
    output_examples_path: Optional[str] = None,
):
    """Test SFT policy using all geometrically valid masks (not just legal-only) on dataset grids
    
    Args:
        collect_examples: If True, emit one corrective (legal) example for every grid that fails
        output_examples_path: Path to output JSONL file with collected corrective examples
    
    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # load policy
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint)
    policy.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Testing on {len(grids)} grids with max {max_moves_per_grid} moves per grid\n")
    
    # Aggregate statistics
    all_grid_results = []
    total_moves = 0
    total_valid_moves = 0
    total_reward = 0
    
    # For collecting corrective training examples (one per failed grid)
    corrective_examples = []
    seen_failure_grids: Set[str] = set()  # Track unique grids only
    
    if not grids:
        raise ValueError("No grids provided for evaluation.")
    
    # Use tqdm for progress bar
    for grid_idx, initial_grid in enumerate(tqdm(grids, desc="Testing grids", unit="grid")):
        # create environments
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
        first_invalid_move = None  # Track first invalid move for diagnostics
        executed_moves = []  # Track executed moves (r1, c1, r2, c2) for state replay
        
        for move_num in range(max_moves_per_grid):
            # phase-0: use ALL geometrically valid anchors (not just legal ones)
            phase0_obs = obs.unsqueeze(0).to(device)
            phase0_mask = wrapped_env.get_action_mask()  # ALL geometrically valid anchors
            
            # pad to 170 if needed
            if phase0_mask.shape[0] < 170:
                padded = torch.zeros(170, dtype=torch.bool)
                padded[:phase0_mask.shape[0]] = phase0_mask
                phase0_mask = padded
            phase0_mask = phase0_mask.unsqueeze(0).to(device)
            
            if phase0_mask.sum() == 0:
                break
            
            # Get top-3 anchor choices from policy
            with torch.no_grad():
                logits, _, _ = policy(phase0_obs, phase0_mask)  # ignore value and sum_predictions
                valid_indices = torch.nonzero(phase0_mask[0], as_tuple=False).squeeze(-1)
                if valid_indices.numel() == 0:
                    break
                valid_logits = logits[0][valid_indices]
                # Get top-3 choices (or fewer if less than 3 available)
                k = min(3, valid_indices.numel())
                topk_values, topk_indices_compact = torch.topk(valid_logits, k)
                top_anchor_indices = [valid_indices[topk_indices_compact[i]].item() for i in range(k)]
            
            # Try all 9 combinations: top-3 anchors × top-3 extents for each anchor
            move_success = False
            total_attempts = 0
            
            for anchor_attempt_idx, anchor_idx in enumerate(top_anchor_indices):
            r1, c1 = flat_idx_to_anchor(anchor_idx)
                
                # Save state before trying this anchor - recreate wrapped_env from current validation_env state
                wrapped_env_save = TwoPhaseWrapper(
                    Sum10GymEnv(initial_grid=validation_env.grid.copy()),
                    curriculum_legal_only=False,
                    curriculum_updates=0
                )
                wrapped_env_save.reset()
                # Replay all previous valid moves to get back to current state
                for prev_r1, prev_c1, prev_r2, prev_c2 in executed_moves:
                    anchor_idx_prev = prev_r1 * 17 + prev_c1
                    wrapped_env_save.step(anchor_idx_prev)  # Phase-0
                    # Find extent index for phase-1
                    width = 17 - prev_c1
                    extent_idx_prev = (prev_r2 - prev_r1) * width + (prev_c2 - prev_c1)
                    wrapped_env_save.step(extent_idx_prev)  # Phase-1
            
            # step Phase-0
                obs_after_anchor, reward, terminated, truncated, info = wrapped_env_save.step(anchor_idx)
            
            # phase-1: use ALL geometrically valid extents (not just legal ones)
                phase1_obs = obs_after_anchor.unsqueeze(0).to(device)
                phase1_mask = wrapped_env_save.get_action_mask()  # ALL geometrically valid extents
            
            # pad to 170 if needed
            if phase1_mask.shape[0] < 170:
                padded = torch.zeros(170, dtype=torch.bool)
                padded[:phase1_mask.shape[0]] = phase1_mask
                phase1_mask = padded
            phase1_mask = phase1_mask.unsqueeze(0).to(device)
            
            if phase1_mask.sum() == 0:
                    # No valid extents for this anchor, try next anchor
                    total_attempts += 1
                    continue
            
                # Get top-3 extent choices from policy
            with torch.no_grad():
                    logits, _, _ = policy(phase1_obs, phase1_mask)  # ignore value and sum_predictions
                valid_indices = torch.nonzero(phase1_mask[0], as_tuple=False).squeeze(-1)
                if valid_indices.numel() == 0:
                        # No valid extents, try next anchor
                        total_attempts += 1
                        continue
                valid_logits = logits[0][valid_indices]
                    # Get top-3 choices (or fewer if less than 3 available)
                    k = min(3, valid_indices.numel())
                    topk_values, topk_indices_compact = torch.topk(valid_logits, k)
                    top_extent_indices = [valid_indices[topk_indices_compact[i]].item() for i in range(k)]
                
                # Try all 3 extent choices for this anchor
                for extent_attempt_idx, extent_idx in enumerate(top_extent_indices):
            r2, c2 = flat_idx_to_extent(r1, c1, extent_idx)
            
                    # Use validation_env directly instead of creating a new one each time
                    # We need to check if the move is valid without modifying validation_env
                    # So we'll use a temporary copy only when needed
                    # For now, use the validation_env's step method which validates without side effects
                    # Actually, step() modifies the env, so we need a temp copy
                    # But we can optimize by reusing the same temp env for all extent attempts
                    if extent_attempt_idx == 0:
                        # Create temp env only once per anchor attempt
                        temp_validation_env = Sum10Env()
                        temp_validation_env.reset(grid=initial_grid.copy())
                        # Replay all previous valid moves
                        for prev_r1, prev_c1, prev_r2, prev_c2 in executed_moves:
                            temp_validation_env.step(prev_r1, prev_c1, prev_r2, prev_c2)
                    
                    # For subsequent extent attempts, we need to reset to the same state
                    # Actually, we can't reuse since step() modifies state
                    # So we need to recreate for each attempt, but we can optimize the replay
                    if extent_attempt_idx > 0:
                        # Reset and replay for subsequent attempts
                        temp_validation_env.reset(grid=initial_grid.copy())
                        for prev_r1, prev_c1, prev_r2, prev_c2 in executed_moves:
                            temp_validation_env.step(prev_r1, prev_c1, prev_r2, prev_c2)
                    
                    # validate move
                    step_info = temp_validation_env.step(r1, c1, r2, c2)
            is_valid = step_info.valid
                    move_reward = step_info.reward if is_valid else 0
                    
                    total_attempts += 1
                    
                    if is_valid:
                        # Valid move found! Execute it and break out of retry loops
                        move_success = True
                        
                        # Track first invalid move for diagnostics (if we had retries)
                        if anchor_attempt_idx > 0 or extent_attempt_idx > 0:
                            if first_invalid_move is None:
                                first_invalid_move = {
                                    'move_num': move_num + 1,
                                    'anchor': (r1, c1),
                                    'extent': (r2, c2),
                                    'retries': anchor_attempt_idx + extent_attempt_idx,
                                }
                        
                        grid_moves += 1
            total_moves += 1
                        grid_valid_moves += 1
                        total_valid_moves += 1
                        grid_reward += move_reward
                        total_reward += move_reward
                        grid_rewards_per_move.append(move_reward)
                        executed_moves.append((r1, c1, r2, c2))
                        
                        # Update validation_env
                        validation_env.step(r1, c1, r2, c2)
            
            # Step Phase-1 in wrapped_env (this actually executes the move and updates state)
                        obs, reward, terminated, truncated, info = wrapped_env_save.step(extent_idx)
                        wrapped_env = wrapped_env_save  # Use the successful state
                        
                        if terminated:
                            grid_terminated = True
                        if truncated:
                            grid_truncated = True
                        break
                    else:
                        # Invalid extent, try next extent choice
                        # Track first invalid move for diagnostics
                        if first_invalid_move is None:
                            first_invalid_move = {
                                'move_num': move_num + 1,
                                'anchor': (r1, c1),
                                'extent': (r2, c2),
                                'retries': anchor_attempt_idx + extent_attempt_idx,
                            }
                
                if move_success:
                    break
            
            # If we exhausted all 9 attempts (3 anchors × 3 extents), end the grid
            if not move_success:
                # All 9 combinations failed - count this as ONE invalid move attempt, then end the grid
                grid_moves += 1
                total_moves += 1
                
                # Collect corrective example if requested (one per failed grid)
                if collect_examples:
                    grid_at_failure = validation_env.grid.copy()
                    grid_signature = get_grid_hash(grid_at_failure)
                    if grid_signature not in seen_failure_grids:
                        legal_moves = validation_env.enumerate_legal()
                        if legal_moves:
                            best_move, best_reward = max(legal_moves, key=lambda item: item[1])
                            (r1_best, c1_best, r2_best, c2_best) = best_move
                            corrective_examples.append({
                                "episode_id": f"test_grid_{grid_idx}_correction_seed{100000 + grid_idx}",
                                "step": move_num + 1,
                                "grid": grid_at_failure.tolist(),
                                "action": {"r1": r1_best, "c1": c1_best, "r2": r2_best, "c2": c2_best},
                                "num_legal_actions": len(legal_moves),
                                "legal": True,
                                "reward": int(best_reward),
                                "done": False,
                                "agent_tag": "sft-correction",
                                "rng_seed": 100000 + grid_idx,
                            })
                            seen_failure_grids.add(grid_signature)
                        else:
                            tqdm.write(f"[WARN] No legal moves found at failure state for grid {grid_idx + 1}")
                
                # Track first invalid move for diagnostics if not already tracked
                if first_invalid_move is None:
                    # Use the last attempted move as the invalid move
                    if total_attempts > 0:
                        # We tried moves but all failed, mark as invalid
                        first_invalid_move = {
                            'move_num': move_num + 1,
                            'anchor': None,  # Couldn't find a valid move
                            'extent': None,
                            'retries': total_attempts - 1,  # Number of retries before giving up
                        }
                break
            
            if grid_terminated or grid_truncated:
                break
        
        # Store per-grid results
        grid_legality_rate = (grid_valid_moves / grid_moves * 100) if grid_moves > 0 else 0.0
        avg_reward_per_move = grid_reward / grid_valid_moves if grid_valid_moves > 0 else 0.0
        avg_reward_per_all_moves = grid_reward / grid_moves if grid_moves > 0 else 0.0
        
        grid_result = {
            'grid_idx': grid_idx,
            'num_moves': grid_moves,
            'num_valid_moves': grid_valid_moves,
            'legality_rate': grid_legality_rate,
            'total_reward': grid_reward,
            'avg_reward_per_valid_move': avg_reward_per_move,
            'avg_reward_per_all_moves': avg_reward_per_all_moves,
            'terminated': grid_terminated,
            'truncated': grid_truncated,
            'rewards_per_move': grid_rewards_per_move,
            'first_invalid_move': first_invalid_move,
            'executed_moves': executed_moves,  # Store for example collection
        }
        all_grid_results.append(grid_result)
        
        if verbose:
            tqdm.write(f"Grid {grid_idx + 1}: moves={grid_moves}, valid={grid_valid_moves}, "
                      f"legality={grid_legality_rate:.1f}%, reward={grid_reward}, "
                      f"avg_reward/valid={avg_reward_per_move:.2f}, "
                      f"terminated={grid_terminated}, truncated={grid_truncated}")
    
    # Compute aggregate statistics
    overall_legality_rate = (total_valid_moves / total_moves * 100) if total_moves > 0 else 0.0
    
    legality_rates = [r['legality_rate'] for r in all_grid_results]
    total_rewards = [r['total_reward'] for r in all_grid_results]
    num_moves_list = [r['num_moves'] for r in all_grid_results]
    num_valid_moves_list = [r['num_valid_moves'] for r in all_grid_results]
    avg_reward_per_valid_list = [r['avg_reward_per_valid_move'] for r in all_grid_results if r['num_valid_moves'] > 0]
    avg_reward_per_all_list = [r['avg_reward_per_all_moves'] for r in all_grid_results]
    
    terminated_count = sum(1 for r in all_grid_results if r['terminated'])
    truncated_count = sum(1 for r in all_grid_results if r['truncated'])
    
    # Analyze early termination patterns
    early_terminated = [r for r in all_grid_results if r['terminated'] and r['num_moves'] <= 5]
    grids_with_invalid_moves = [r for r in all_grid_results if r['first_invalid_move'] is not None]
    
    # Print comprehensive results
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE RESULTS WITH ALL GEOMETRICALLY VALID MASKS")
    print(f"{'='*70}")
    print(f"\nOverall Statistics:")
    print(f"  Total grids tested: {len(grids)}")
    print(f"  Total moves: {total_moves}")
    print(f"  Total valid moves: {total_valid_moves}")
    print(f"  Overall legality rate: {overall_legality_rate:.2f}%")
    print(f"  Total cells cleared: {total_reward}")
    print(f"  Average cells cleared per grid: {np.mean(total_rewards):.2f}")
    
    print(f"\nLegality Rate Statistics:")
    print(f"  Mean: {np.mean(legality_rates):.2f}%")
    print(f"  Median: {np.median(legality_rates):.2f}%")
    print(f"  Min: {np.min(legality_rates):.2f}%")
    print(f"  Max: {np.max(legality_rates):.2f}%")
    print(f"  Std: {np.std(legality_rates):.2f}%")
    
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
    
    print(f"\nValid Moves per Grid:")
    print(f"  Mean: {np.mean(num_valid_moves_list):.2f}")
    print(f"  Median: {np.median(num_valid_moves_list):.2f}")
    print(f"  Min: {np.min(num_valid_moves_list)}")
    print(f"  Max: {np.max(num_valid_moves_list)}")
    print(f"  Std: {np.std(num_valid_moves_list):.2f}")
    
    if avg_reward_per_valid_list:
        print(f"\nAverage Cells Cleared per Valid Move:")
        print(f"  Mean: {np.mean(avg_reward_per_valid_list):.2f}")
        print(f"  Median: {np.median(avg_reward_per_valid_list):.2f}")
        print(f"  Min: {np.min(avg_reward_per_valid_list):.2f}")
        print(f"  Max: {np.max(avg_reward_per_valid_list):.2f}")
        print(f"  Std: {np.std(avg_reward_per_valid_list):.2f}")
    
    print(f"\nAverage Cells Cleared per All Moves (including invalid):")
    print(f"  Mean: {np.mean(avg_reward_per_all_list):.2f}")
    print(f"  Median: {np.median(avg_reward_per_all_list):.2f}")
    print(f"  Min: {np.min(avg_reward_per_all_list):.2f}")
    print(f"  Max: {np.max(avg_reward_per_all_list):.2f}")
    print(f"  Std: {np.std(avg_reward_per_all_list):.2f}")
    
    print(f"\nEpisode Completion:")
    print(f"  Terminated (solved): {terminated_count} ({terminated_count/len(grids)*100:.1f}%)")
    print(f"  Truncated (max moves): {truncated_count} ({truncated_count/len(grids)*100:.1f}%)")
    print(f"  Incomplete: {len(grids) - terminated_count - truncated_count} ({(len(grids) - terminated_count - truncated_count)/len(grids)*100:.1f}%)")
    
    # Early termination analysis
    if early_terminated:
        print(f"\nEarly Termination Analysis (≤5 moves):")
        print(f"  Grids terminated early: {len(early_terminated)} ({len(early_terminated)/len(grids)*100:.1f}%)")
        avg_legality_early = np.mean([r['legality_rate'] for r in early_terminated])
        print(f"  Average legality rate for early-terminated grids: {avg_legality_early:.1f}%")
        early_with_invalid = [r for r in early_terminated if r['first_invalid_move'] is not None]
        if early_with_invalid:
            avg_first_invalid_move = np.mean([r['first_invalid_move']['move_num'] for r in early_with_invalid])
            print(f"  Average move number of first invalid move: {avg_first_invalid_move:.1f}")
            print(f"  Note: Early termination often occurs after an invalid move")
    
    # Invalid move analysis
    if grids_with_invalid_moves:
        print(f"\nInvalid Move Analysis:")
        print(f"  Grids with at least one invalid move: {len(grids_with_invalid_moves)} ({len(grids_with_invalid_moves)/len(grids)*100:.1f}%)")
        first_invalid_move_nums = [r['first_invalid_move']['move_num'] for r in grids_with_invalid_moves]
        print(f"  First invalid move statistics:")
        print(f"    Mean: {np.mean(first_invalid_move_nums):.1f}")
        print(f"    Median: {np.median(first_invalid_move_nums):.1f}")
        print(f"    Min: {np.min(first_invalid_move_nums)}")
        print(f"    Max: {np.max(first_invalid_move_nums)}")
        # Count how many had invalid move in first 3 moves
        early_invalid = sum(1 for num in first_invalid_move_nums if num <= 3)
        print(f"  Invalid moves in first 3 moves: {early_invalid} ({early_invalid/len(grids_with_invalid_moves)*100:.1f}% of grids with invalid moves)")
    
    # Distribution analysis
    print(f"\nDistribution of Total Cells Cleared:")
    reward_dist = Counter([int(r) for r in total_rewards])
    for reward_val in sorted(reward_dist.keys())[:10]:  # Show top 10 most common
        print(f"  {reward_val} cells: {reward_dist[reward_val]} grids")
    if len(reward_dist) > 10:
        print(f"  ... ({len(reward_dist) - 10} more unique values)")
    
    print(f"\n{'='*70}")
    print(f"This test verifies if the SFT policy learned to avoid illegal actions")
    print(f"when they're present in the mask (as it was trained).")
    print(f"{'='*70}\n")
    
    # Collect and save corrective examples if requested
    if collect_examples and output_examples_path:
        if corrective_examples:
            print(f"\nWriting {len(corrective_examples)} corrective examples to {output_examples_path}")
            with open(output_examples_path, 'w') as f:
                for example in corrective_examples:
                    f.write(json.dumps(example) + '\n')
            print("  Done.")
        else:
            print("\nNo corrective examples collected; skipping write.")
    
    # Return comprehensive results
    results = {
        'overall_legality_rate': overall_legality_rate,
        'total_moves': total_moves,
        'total_valid_moves': total_valid_moves,
        'total_reward': total_reward,
        'num_grids': len(grids),
        'legality_rate_stats': {
            'mean': float(np.mean(legality_rates)),
            'median': float(np.median(legality_rates)),
            'min': float(np.min(legality_rates)),
            'max': float(np.max(legality_rates)),
            'std': float(np.std(legality_rates)),
        },
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SFT checkpoint or wandb artifact")
    parser.add_argument("--num_grids", type=int, default=50, help="Number of grids to test (default: 50)")
    parser.add_argument("--dataset_name", type=str, default="djdumpling/fruit-box-minimal-area", help="Dataset to load via fruit_box loader")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split")
    parser.add_argument("--loader_seed", type=int, default=None, help="Seed passed to fruit_box.load_environment")
    parser.add_argument("--max_moves", type=int, default=60, help="Maximum moves per grid (default: 60)")
    parser.add_argument("--verbose", action="store_true", help="Print per-grid statistics")
    parser.add_argument("--collect_examples", action="store_true", help="Collect failed and successful examples for training")
    parser.add_argument("--output_examples", type=str, default=None, help="Output JSONL file path for collected examples")
    args = parser.parse_args()
    
    # handle wandb artifact download if needed
    checkpoint_path = args.checkpoint
    if args.checkpoint.startswith("djdumpling") or ("/" in args.checkpoint and ":" in args.checkpoint):
        # Looks like a wandb artifact path
        checkpoint_path = load_checkpoint_from_wandb(args.checkpoint)
    
    grids = load_grids_from_loader(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        num_grids=args.num_grids,
        seed=args.loader_seed,
    )
    
    results = test_policy_with_all_masks(
        checkpoint_path, 
        grids,
        max_moves_per_grid=args.max_moves,
        verbose=args.verbose,
        collect_examples=args.collect_examples,
        output_examples_path=args.output_examples,
    )

