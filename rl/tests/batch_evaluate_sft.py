"""Batch evaluation of SFT policy on multiple random grids"""
import sys
from pathlib import Path
# Add project root to path (go up 2 levels from rl/tests/batch_evaluate_sft.py)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import numpy as np
import torch
from typing import Dict, List, Tuple
from collections import Counter

from rl.models.policy import CNNPolicy
from rl.envs.sum10_env import Sum10GymEnv
from rl.envs.split_wrapper import TwoPhaseWrapper
from fruit_box import Sum10Env
from rl.train_sft import flat_idx_to_anchor, flat_idx_to_extent


def generate_random_grid(seed: int) -> np.ndarray:
    """Generate a random 10x17 grid with values 1-9"""
    rng = np.random.default_rng(seed)
    grid = rng.integers(1, 10, size=(10, 17), dtype=np.uint8)
    return grid


def run_policy_on_grid(
    policy: CNNPolicy,
    initial_grid: np.ndarray,
    max_moves: int,
    device: torch.device,
    grid_seed: int,
) -> Dict:
    """Run policy on a grid and return stats (legality, cells cleared, etc)"""
    # create environment
    env = Sum10GymEnv(initial_grid=initial_grid.copy())
    wrapped_env = TwoPhaseWrapper(env, curriculum_legal_only=False, curriculum_updates=0)
    
    # create Sum10Env for validation
    validation_env = Sum10Env()
    validation_env.reset(grid=initial_grid.copy())
    
    # statistics
    all_rewards = []
    all_valid = []
    total_moves = 0
    valid_moves = 0
    
    # run policy
    obs, info = wrapped_env.reset()
    
    for move_num in range(max_moves):
        # phase-0: select anchor
        phase0_obs = obs.unsqueeze(0).to(device)  # [1, 4, 10, 17]
        
        # find all anchors that have at least one legal extent
        legal_anchors_set = set()
        for anchor_r1 in range(10):
            for anchor_c1 in range(17):
                anchor_idx = anchor_r1 * 17 + anchor_c1
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
                    legal_anchors_set.add(anchor_idx)
        
        # build mask: True only at positions corresponding to legal anchors
        phase0_mask = torch.zeros(170, dtype=torch.bool)
        for legal_anchor_idx in sorted(legal_anchors_set):
            phase0_mask[legal_anchor_idx] = True
        phase0_mask = phase0_mask.unsqueeze(0).to(device)  # [1, 170]
        
        if phase0_mask.sum() == 0:
            # no legal anchors available
            break
        
        with torch.no_grad():
            logits, _ = policy(phase0_obs, phase0_mask)
            # extract logits only at legal anchor positions
            legal_anchor_indices = torch.nonzero(phase0_mask[0], as_tuple=False).squeeze(-1)
            valid_logits = logits[0][legal_anchor_indices]
            anchor_idx_compact = valid_logits.argmax().item()
            anchor_idx = legal_anchor_indices[anchor_idx_compact].item()
        
        r1, c1 = flat_idx_to_anchor(anchor_idx)
        
        # step Phase-0
        obs, reward, terminated, truncated, info = wrapped_env.step(anchor_idx)
        
        # phase-1: use legal-only mask (only extents that sum to 10)
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
            else:
                # no legal moves available
                break
        
        r2, c2 = flat_idx_to_extent(r1, c1, extent_idx)
        
        # validate move using Sum10Env
        step_info = validation_env.step(r1, c1, r2, c2)
        
        is_valid = step_info.valid
        reward_value = step_info.reward if is_valid else 0
        
        # step Phase-1 in wrapped env
        obs, reward, terminated, truncated, info = wrapped_env.step(extent_idx)
        
        # collect statistics
        all_rewards.append(reward_value)
        all_valid.append(is_valid)
        total_moves += 1
        if is_valid:
            valid_moves += 1
        
        # check if episode ended
        if terminated or truncated:
            break
    
    # calculate statistics
    legality_rate = valid_moves / max(total_moves, 1)
    total_cells_cleared = sum(all_rewards)
    
    return {
        'legality_rate': legality_rate,
        'total_cells_cleared': total_cells_cleared,
        'num_moves': total_moves,
        'num_valid_moves': valid_moves,
        'reward_per_move': all_rewards,
        'grid_seed': grid_seed,
    }


def load_checkpoint_from_wandb(artifact_path: str) -> str:
    """Download checkpoint from wandb artifact and return local path.
    
    Args:
        artifact_path: Wandb artifact path (e.g., 'djdumpling-yale/fruit-box-sft/sft-checkpoint-epoch-160:v0')
    
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


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate SFT policy on random grids")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path or wandb artifact")
    parser.add_argument("--wandb-artifact", action="store_true", help="Download checkpoint from wandb")
    parser.add_argument("--num_grids", type=int, default=30, help="Number of grids to evaluate")
    parser.add_argument("--max_moves", type=int, default=50, help="Max moves per grid")
    parser.add_argument("--seed_start", type=int, default=10000, help="Starting seed (default: 10000 to avoid training data)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto if not specified)")
    parser.add_argument("--output", type=str, default=None, help="Output path for episodes.jsonl")
    parser.add_argument("--agent_tag", type=str, default="sft", help="Agent tag in episodes.jsonl")
    
    args = parser.parse_args()
    
    # handle wandb artifact download if needed
    checkpoint_path = args.checkpoint
    if args.wandb_artifact or (args.checkpoint.startswith("djdumpling") or "/" in args.checkpoint and ":" in args.checkpoint):
        # Looks like a wandb artifact path
        checkpoint_path = load_checkpoint_from_wandb(args.checkpoint)
    
    # setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # load policy
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy.eval()
    print("Policy loaded successfully!")
    
    print(f"\n{'='*70}")
    print(f"EVALUATING ON {args.num_grids} RANDOM GRIDS")
    print(f"{'='*70}")
    print(f"Max moves per grid: {args.max_moves}")
    print(f"Seed range: {args.seed_start} to {args.seed_start + args.num_grids - 1}")
    print()
    
    # run evaluation on each grid
    all_results = []
    for i in range(args.num_grids):
        grid_seed = args.seed_start + i
        initial_grid = generate_random_grid(seed=grid_seed)
        
        result = run_policy_on_grid(
            policy=policy,
            initial_grid=initial_grid,
            max_moves=args.max_moves,
            device=device,
            grid_seed=grid_seed,
        )
        all_results.append(result)
        
        # print progress
        print(f"Grid {i+1}/{args.num_grids} (seed={grid_seed}): "
              f"legality={result['legality_rate']:.1%}, "
              f"moves={result['num_moves']}, "
              f"total_cleared={result['total_cells_cleared']}")
    
    # aggregate statistics
    print(f"\n{'='*70}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*70}")
    
    # legality rate
    all_legality_rates = [r['legality_rate'] for r in all_results]
    overall_legality = np.mean(all_legality_rates)
    print(f"\nOverall Legality Rate:")
    print(f"  Mean: {overall_legality:.2%}")
    print(f"  Min: {min(all_legality_rates):.2%}")
    print(f"  Max: {max(all_legality_rates):.2%}")
    print(f"  Std: {np.std(all_legality_rates):.2%}")
    
    # total cells cleared per grid
    all_total_cleared = [r['total_cells_cleared'] for r in all_results]
    print(f"\nTotal Cells Cleared per Grid:")
    print(f"  Mean: {np.mean(all_total_cleared):.1f} cells")
    print(f"  Median: {np.median(all_total_cleared):.1f} cells")
    print(f"  Min: {min(all_total_cleared)} cells")
    print(f"  Max: {max(all_total_cleared)} cells")
    print(f"  Std: {np.std(all_total_cleared):.1f} cells")
    
    # distribution of total cells cleared
    cleared_distribution = Counter(all_total_cleared)
    print(f"\nDistribution of Total Cells Cleared:")
    print(f"  (showing all unique values)")
    for cleared_count in sorted(cleared_distribution.keys()):
        count = cleared_distribution[cleared_count]
        percentage = count / len(all_results) * 100
        print(f"  {cleared_count} cells: {count} grids ({percentage:.1f}%)")
    
    # additional statistics
    all_num_moves = [r['num_moves'] for r in all_results]
    all_num_valid = [r['num_valid_moves'] for r in all_results]
    
    print(f"\nMove Statistics:")
    print(f"  Average moves per grid: {np.mean(all_num_moves):.1f}")
    print(f"  Average valid moves per grid: {np.mean(all_num_valid):.1f}")
    
    # reward per move statistics (across all grids)
    all_rewards = []
    for r in all_results:
        all_rewards.extend(r['reward_per_move'])
    
    if all_rewards:
        print(f"\nReward per Move (across all grids):")
        print(f"  Average: {np.mean(all_rewards):.2f} cells")
        print(f"  Median: {np.median(all_rewards):.2f} cells")
        print(f"  Min: {min(all_rewards)} cells")
        print(f"  Max: {max(all_rewards)} cells")
        print(f"  Std: {np.std(all_rewards):.2f} cells")
        
        # reward distribution
        reward_dist = Counter(all_rewards)
        print(f"\nReward Distribution (per move):")
        for reward_val in sorted(reward_dist.keys()):
            count = reward_dist[reward_val]
            percentage = count / len(all_rewards) * 100
            print(f"  {reward_val} cells: {count} moves ({percentage:.1f}%)")
    
    # summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Evaluated {args.num_grids} random grids")
    print(f"Overall legality rate: {overall_legality:.2%}")
    print(f"Average cells cleared per grid: {np.mean(all_total_cleared):.1f}")
    print(f"Total cells cleared across all grids: {sum(all_total_cleared)}")
    
    # export episodes.jsonl if output path is specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        episodes = []
        for result in all_results:
            episode = {
                "episode_id": f"seed{result['grid_seed']}",
                "seed": result['grid_seed'],
                "agent_tag": args.agent_tag,
                "total_reward": int(result['total_cells_cleared']),
                "total_steps": int(result['num_moves'])
            }
            episodes.append(episode)
        
        with open(output_path, 'w') as f:
            for episode in episodes:
                f.write(json.dumps(episode) + '\n')
        
        print(f"\nExported {len(episodes)} episodes to: {output_path}")

if __name__ == "__main__":
    main()