# === MINIMAL_AREA ===
# Total Reward: 113.72 ± 14.89 (min: 65, max: 154)
# Total Steps: 51.44 ± 6.61 (min: 31, max: 70)
# Reward per Turn: 2.21 ± 0.06

# === HIGH_VALUE_PAIRS ===
# Total Reward: 113.73 ± 15.02 (min: 65, max: 161)
# Total Steps: 51.42 ± 6.68 (min: 31, max: 72)
# Reward per Turn: 2.21 ± 0.06

# === WIN RATE ANALYSIS ===

# Head-to-Head Results (1000 seeds):
# Minimal Area wins: 294 (29.4%)
# High Value Pairs wins: 255 (25.5%)
# Ties: 451 (45.1%)

# === SUMMARY ===
# Minimal area wins (294 vs 255)

import json
import pandas as pd
from pathlib import Path

def load_episodes(filepath: Path) -> pd.DataFrame:
    """Load episodes from JSONL file."""
    episodes = []
    with open(filepath, 'r') as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    return pd.DataFrame(episodes)

def compare_policies():
    base_dir = Path("out_data")
    
    minimal_data = load_episodes(base_dir / "minimal_area_1k" / "episodes.jsonl")
    high_pairs_data = load_episodes(base_dir / "high_pairs_1k" / "episodes.jsonl")
    
    
    policies = {
        "minimal_area": minimal_data,
        "high_value_pairs": high_pairs_data
    }
    
    results = {}
    for policy_name, df in policies.items():
        total_reward = df['total_reward']
        total_steps = df['total_steps']
        reward_per_turn = total_reward / total_steps
        
        results[policy_name] = {
            'total_reward': total_reward,
            'total_steps': total_steps,
            'reward_per_turn': reward_per_turn
        }
        
        print(f"=== {policy_name.upper()} ===")
        print(f"Total Reward: {total_reward.mean():.2f} ± {total_reward.std():.2f} (min: {total_reward.min()}, max: {total_reward.max()})")
        print(f"Total Steps: {total_steps.mean():.2f} ± {total_steps.std():.2f} (min: {total_steps.min()}, max: {total_steps.max()})")
        print(f"Reward per Turn: {reward_per_turn.mean():.2f} ± {reward_per_turn.std():.2f}")
        print()
    
    print("=== WIN RATE ANALYSIS ===")
    
    minimal_wins = 0
    high_pairs_wins = 0
    ties = 0
    
    for seed in range(1, 1001):
        minimal_reward = minimal_data[minimal_data['seed'] == seed]['total_reward'].iloc[0]
        high_pairs_reward = high_pairs_data[high_pairs_data['seed'] == seed]['total_reward'].iloc[0]
        
        if minimal_reward > high_pairs_reward:
            minimal_wins += 1
        elif high_pairs_reward > minimal_reward:
            high_pairs_wins += 1
        else:
            ties += 1
    
    total_comparisons = 1000
    print(f"\nHead-to-Head Results ({total_comparisons} seeds):")
    print(f"Minimal Area wins: {minimal_wins} ({minimal_wins/total_comparisons*100:.1f}%)")
    print(f"High Value Pairs wins: {high_pairs_wins} ({high_pairs_wins/total_comparisons*100:.1f}%)")
    print(f"Ties: {ties} ({ties/total_comparisons*100:.1f}%)")
    
    print("\n=== SUMMARY ===")
    if minimal_wins > high_pairs_wins:
        print(f"Minimal area wins ({minimal_wins} vs {high_pairs_wins})")
    elif high_pairs_wins > minimal_wins:
        print(f"High value pairs win ({high_pairs_wins} vs {minimal_wins})")
    else:
        print(f"Tie ({minimal_wins} vs {high_pairs_wins})")

if __name__ == "__main__":
    compare_policies()
