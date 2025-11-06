"""Evaluation script for trained GRPO policy."""
import argparse
import numpy as np
import torch
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm
from datasets import load_dataset

from rl.envs.sum10_env import Sum10GymEnv
from rl.envs.split_wrapper import TwoPhaseWrapper
from rl.models.policy import CNNPolicy
from fruit_box import Sum10Env


def compute_oracle_regret(env: Sum10Env, actions_taken: List[Tuple]) -> float:
    """Compute depth-1 oracle regret.
    
    Compares actual actions taken to best immediate move at each step.
    
    Args:
        env: Sum10Env instance
        actions_taken: List of (r1, c1, r2, c2) tuples
    
    Returns:
        regret: Average regret per step
    """
    total_regret = 0.0
    valid_steps = 0
    
    # reset env to initial state
    initial_grid = env.grid.copy()
    test_env = Sum10Env()
    test_env.reset(grid=initial_grid)
    
    for action in actions_taken:
        # get best immediate move
        legal_moves = test_env.enumerate_legal()
        if not legal_moves:
            break
        
        best_reward = max(reward for _, reward in legal_moves)
        
        # execute actual action
        r1, c1, r2, c2 = action
        step_info = test_env.step(r1, c1, r2, c2)
        
        if step_info.valid:
            actual_reward = step_info.reward
            regret = best_reward - actual_reward
            total_regret += regret
            valid_steps += 1
        else:
            break
    
    return total_regret / max(valid_steps, 1)


def evaluate(
    policy: CNNPolicy,
    env_factory,
    num_episodes: int = 100,
    seeds: Optional[List[int]] = None,
) -> Dict:
    """Evaluate policy.
    
    Args:
        policy: Trained policy network
        env_factory: Function that creates environment
        num_episodes: Number of episodes to evaluate
        seeds: Optional list of seeds for deterministic evaluation
    
    Returns:
        Dictionary with evaluation metrics
    """
    device = next(policy.parameters()).device
    policy.eval()
    
    total_rewards = []
    legality_rates = []
    cells_cleared_per_move = []
    oracle_regrets = []
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        seed = seeds[episode] if seeds and episode < len(seeds) else None
        env = env_factory(seed)
        
        obs, info = env.reset()
        # store initial grid for oracle regret computation
        initial_grid = env.game_env.grid.copy() if hasattr(env, 'game_env') else None
        obs = obs.unsqueeze(0).to(device)  # [1, 4, 10, 17]
        
        episode_reward = 0.0
        valid_moves = 0
        total_moves = 0
        actions_taken = []  # store (r1, c1, r2, c2) tuples
        last_anchor = None
        
        max_steps = 200  # safety limit
        step_count = 0
        
        while step_count < max_steps:
            # get action mask
            mask = env.get_action_mask().unsqueeze(0).to(device)  # [1, action_dim]
            
            # select action
            with torch.no_grad():
                action, _, _ = policy.get_action_and_value(obs, mask)
            
            action_idx = action[0].item()
            
            # track action before step (for Phase-1)
            if env.phase == 1 and env.selected_anchor is not None:
                r1, c1 = env.selected_anchor
                r2, c2 = env.flat_idx_to_extent(r1, c1, action_idx)
                last_anchor = (r1, c1, r2, c2)
            
            # step environment
            obs_new, reward, terminated, truncated, info = env.step(action_idx)
            obs_new = obs_new.unsqueeze(0).to(device)
            
            episode_reward += reward
            total_moves += 1
            
            # track valid moves and actions
            if reward > 0:
                valid_moves += 1
                if last_anchor:
                    actions_taken.append(last_anchor)
            
            obs = obs_new
            step_count += 1
            
            if terminated or truncated:
                break
        
        # compute oracle regret if we have actions and initial grid
        if actions_taken and initial_grid is not None:
            test_env = Sum10Env()
            test_env.reset(grid=initial_grid.copy())
            oracle_regret = compute_oracle_regret(test_env, actions_taken)
            oracle_regrets.append(oracle_regret)
        else:
            oracle_regrets.append(0.0)  # no valid actions, no regret
        
        total_rewards.append(episode_reward)
        legality_rate = valid_moves / max(total_moves, 1) if total_moves > 0 else 0.0
        legality_rates.append(legality_rate)
        
        if valid_moves > 0:
            avg_cleared = episode_reward / valid_moves
            cells_cleared_per_move.append(avg_cleared)
    
    # compute statistics
    results = {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_legality_rate": np.mean(legality_rates),
        "std_legality_rate": np.std(legality_rates),
        "mean_cells_cleared_per_move": np.mean(cells_cleared_per_move) if cells_cleared_per_move else 0.0,
        "std_cells_cleared_per_move": np.std(cells_cleared_per_move) if cells_cleared_per_move else 0.0,
        "mean_oracle_regret": np.mean(oracle_regrets) if oracle_regrets else 0.0,
    }
    
    return results


def load_grids_from_dataset(
    dataset_name: str = "djdumpling/fruit-box-minimal-area",
    dataset_split: str = "train",
    num_episodes: int = 100,
) -> List[np.ndarray]:
    """Load initial grids from HuggingFace dataset."""
    hf_dataset = load_dataset(dataset_name, split=dataset_split)
    
    # Group by episode_id and get unique initial grids
    episodes = {}
    for row in hf_dataset:
        ep_id = row["episode_id"]
        if ep_id not in episodes:
            episodes[ep_id] = row
    
    # Extract initial grids
    grids = []
    for ep_id, row in list(episodes.items())[:num_episodes]:
        initial_grid = np.array(row["grid"], dtype=np.uint8)
        grids.append(initial_grid)
    
    return grids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random grids")
    parser.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset name (e.g., 'djdumpling/fruit-box-minimal-area')")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use")
    args = parser.parse_args()
    
    # Load policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded policy from {args.checkpoint}")
    
    # Create env factory
    if args.dataset:
        # Load grids from dataset
        print(f"Loading {args.num_episodes} grids from dataset {args.dataset}...")
        grids = load_grids_from_dataset(args.dataset, args.dataset_split, args.num_episodes)
        print(f"Loaded {len(grids)} grids")
        
        def env_factory(seed=None):
            grid_idx = seed if seed is not None and seed < len(grids) else 0
            env = Sum10GymEnv(initial_grid=grids[grid_idx])
            env = TwoPhaseWrapper(env, curriculum_legal_only=False, curriculum_updates=0)
            return env
        
        # Use grid indices as seeds
        seeds = list(range(len(grids)))
    else:
        # Use random grids
        def env_factory(seed=None):
            env = Sum10GymEnv()
            env = TwoPhaseWrapper(env, curriculum_legal_only=False, curriculum_updates=0)
            return env
        
        seeds = None
    
    # Evaluate
    results = evaluate(policy, env_factory, args.num_episodes, seeds=seeds)
    
    # Print results
    print("\n" + "="*70)
    print("Evaluation Results:")
    print("="*70)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Legality Rate: {results['mean_legality_rate']:.3f} ± {results['std_legality_rate']:.3f}")
    print(f"Cells Cleared per Move: {results['mean_cells_cleared_per_move']:.2f} ± {results['std_cells_cleared_per_move']:.2f}")
    print(f"Oracle Regret: {results['mean_oracle_regret']:.3f}")
    print("="*70)


if __name__ == "__main__":
    main()

