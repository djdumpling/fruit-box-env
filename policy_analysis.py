import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict

# styling
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

def load_episodes(filepath):
    episodes = []
    with open(filepath, 'r') as f:
        for line in f:
            episodes.append(json.loads(line))
    return pd.DataFrame(episodes)

def compute_stats(df: pd.DataFrame, metric: str) -> Dict:
    return {
        'mean': df[metric].mean(),
        'std': df[metric].std(),
        'min': df[metric].min(),
        'max': df[metric].max(),
        'median': df[metric].median()
    }

# table-esque formatting for 
def print_statistics_table(all_data: Dict[str, pd.DataFrame]):
    # Calculate reward per turn for each policy
    for policy, df in all_data.items():
        df['reward_per_turn'] = df['total_reward'] / df['total_steps']
    
    print("\nTotal reward statistics")
    print(f"{'Policy':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Median':<12}")
    for policy, df in all_data.items():
        stats = compute_stats(df, 'total_reward')
        print(f"{policy:<20} {stats['mean']:>11.2f} {stats['std']:>11.2f} "
              f"{stats['min']:>11.0f} {stats['max']:>11.0f} {stats['median']:>11.2f}")
    
    print("\nTotal steps statistics")
    print(f"{'Policy':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Median':<12}")
    for policy, df in all_data.items():
        stats = compute_stats(df, 'total_steps')
        print(f"{policy:<20} {stats['mean']:>11.2f} {stats['std']:>11.2f} "
              f"{stats['min']:>11.0f} {stats['max']:>11.0f} {stats['median']:>11.2f}")
    
    print("\nReward/turn statistics")
    print(f"{'Policy':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Median':<12}")
    for policy, df in all_data.items():
        stats = compute_stats(df, 'reward_per_turn')
        print(f"{policy:<20} {stats['mean']:>11.2f} {stats['std']:>11.2f} "
              f"{stats['min']:>11.2f} {stats['max']:>11.2f} {stats['median']:>11.2f}")

def plot_distributions(all_data: Dict[str, pd.DataFrame], output_dir: Path):
    combined = []
    for policy, df in all_data.items():
        temp = df.copy()
        temp['policy'] = policy
        combined.append(temp)
    combined_df = pd.concat(combined, ignore_index=True)
    
    # 1x3 layout
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = sns.color_palette('Set2', len(all_data))
    
    # total reward
    sns.violinplot(data=combined_df, x='policy', y='total_reward', hue='policy', ax=axes[0], palette=colors, inner='box', legend=False)
    axes[0].set_xlabel('Policy', fontsize=12)
    axes[0].set_ylabel('Total Reward', fontsize=12)
    axes[0].set_title('Total Reward Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=15)
    
    # total steps
    sns.violinplot(data=combined_df, x='policy', y='total_steps', hue='policy', ax=axes[1], palette=colors, inner='box', legend=False)
    axes[1].set_xlabel('Policy', fontsize=12)
    axes[1].set_ylabel('Total Steps', fontsize=12)
    axes[1].set_title('Total Steps Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=15)
    
    # reward per turn
    sns.violinplot(data=combined_df, x='policy', y='reward_per_turn', hue='policy', ax=axes[2], palette=colors, inner='box', legend=False)
    axes[2].set_xlabel('Policy', fontsize=12)
    axes[2].set_ylabel('Reward per Turn', fontsize=12)
    axes[2].set_title('Reward per Turn Distribution', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    output_path = output_dir / 'policy_comparisons.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def analyze_seed_performance(all_data: Dict[str, pd.DataFrame]):
    # seeds 1-1000
    seeds = list(range(1, 1001))
    policy_names = list(all_data.keys())
    
    # prep data for each seed
    seed_results = []
    for seed in seeds:
        result = {'seed': seed}
        for policy, df in all_data.items():
            seed_data = df[df['seed'] == seed]
            if not seed_data.empty:
                result[f'{policy}_reward'] = seed_data['total_reward'].values[0]
        seed_results.append(result)
    
    seed_df = pd.DataFrame(seed_results)
    
    # winner for each seed by reward
    def get_winner(row):
        rewards = {p: row[f'{p}_reward'] for p in policy_names if f'{p}_reward' in row}
        return max(rewards, key=rewards.get)
    
    seed_df['winner'] = seed_df.apply(get_winner, axis=1)
    win_counts = seed_df['winner'].value_counts().to_dict()
    
    print("\nWin rate")
    print(f"{'Policy':<20} {'Wins':<10} {'Win %':<10}")
    for policy in policy_names:
        wins = win_counts.get(policy, 0)
        win_pct = (wins / 1000 * 100)
        print(f"{policy:<20} {wins:<10} {win_pct:>8.1f}%")
    print()

def main():
    base_dir = Path("out_data")
    output_dir = Path("out_data/analysis")
    output_dir.mkdir(exist_ok=True)
    
    # policy directories
    policies = {
        'greedy_area': base_dir / 'greedy_area_1k' / 'episodes.jsonl',
        'random_legal': base_dir / 'random_legal_1k' / 'episodes.jsonl',
        'look_ahead': base_dir / 'look_ahead_1k_2_70_0.95' / 'episodes.jsonl',
        'minimal_area': base_dir / 'minimal_area_1k' / 'episodes.jsonl',
        'high_pairs': base_dir / 'high_pairs_1k' / 'episodes.jsonl'
    }
    
    # load data
    all_data = {}
    for policy_name, filepath in policies.items():
        if filepath.exists():
            df = load_episodes(filepath)
            all_data[policy_name] = df
            print(f"Loaded {len(df)} episodes: {policy_name}")
        else:
            print(f"File not found: {filepath}")
    
    print_statistics_table(all_data)
    analyze_seed_performance(all_data)
    plot_distributions(all_data, output_dir)

if __name__ == "__main__":
    main()