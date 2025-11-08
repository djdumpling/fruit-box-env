"""
python rl/train_grpo.py --seed 42

python rl/eval.py \
  --checkpoint checkpoints/policy_final.pt \
  --num_episodes 100 \
  --dataset djdumpling/fruit-box-minimal-area \
  --dataset_split train
"""

import sys
from pathlib import Path
# add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import gymnasium as gym
import wandb

from rl.envs.sum10_env import Sum10GymEnv
from rl.envs.split_wrapper import TwoPhaseWrapper
from rl.models.policy import CNNPolicy
from rl.algo.ppo import compute_gae, compute_ppo_loss
from rl.algo.grpo import compute_grpo_loss, simulate_action_reward
from fruit_box import Sum10Env


@dataclass
class Config:
    """Training configuration."""
    # Data collection
    num_envs: int = 16
    rollout_steps: int = 128
    batch_size: int = 512
    epochs: int = 4 
    
    # Phase-0 (PPO) hyperparameters
    phase0_lr: float = 1e-4
    phase0_clip_eps: float = 0.12
    phase0_target_kl: float = 0.015
    phase0_value_coef: float = 0.8
    
    # Phase-1 (GRPO) hyperparameters
    phase1_lr: float = 3e-4  # reduced from 1e-3 (3x Phase-0 LR instead of 10x) to prevent instability
    phase1_clip_eps: float = 0.2  # reduced from 0.3 for more conservative updates
    grpo_k: int = 16
    grpo_tau: float = 0.7
    frozen_refresh_interval: int = 100  # reduced from 75, less frequent refreshes
    grpo_temperature: float = 1.0
    min_reward_std: float = 0.01
    
    # Shared hyperparameters
    max_updates: int = 2500
    gamma: float = 0.995
    gae_lambda: float = 0.95
    entropy_coef: float = 0.05  # increased from 0.02 for stronger exploration signal
    entropy_target: float = 0.7  # target minimum entropy to prevent collapse
    entropy_penalty_coef: float = 0.2  # strong penalty for entropy below target
    grad_clip: float = 1.0
    lr_warmup_steps: int = 20
    
    # Curriculum learning
    curriculum_updates: int = 500
    illegal_penalty: float = -0.1
    
    # Other
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 500
    render_interval: int = 5
    render_env_idx: int = 0


class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self, rollout_steps: int, num_envs: int, obs_shape: Tuple[int, ...], device: str):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.device = device
        
        # Phase-0 data (anchor selection)
        self.phase0_obs = []
        self.phase0_actions = []
        self.phase0_logprobs = []
        self.phase0_values = []
        self.phase0_rewards = []
        self.phase0_dones = []
        self.phase0_masks = []
        self.phase0_env_indices = []  # track which env each transition belongs to
        
        # Phase-1 data (extent selection)
        self.phase1_obs = []
        self.phase1_anchors = []
        self.phase1_actions = []  # list of [K] arrays
        self.phase1_logprobs = []  # list of [K] arrays
        self.phase1_rewards = []  # list of [K] arrays
        self.phase1_masks = []
        self.phase1_executed_actions = []  # actually executed action
        self.phase1_executed_logprobs = []
        self.phase1_executed_rewards = []
        self.phase1_dones = []
    
    def add_phase0(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        env_idx: int,
    ):
        """Add Phase-0 transition."""
        # detach to avoid double backward
        self.phase0_obs.append(obs.detach().cpu())
        self.phase0_actions.append(action.detach().cpu())
        self.phase0_logprobs.append(logprob.detach().cpu())
        self.phase0_values.append(value.detach().cpu())
        self.phase0_masks.append(mask.detach().cpu())
        self.phase0_env_indices.append(env_idx)
        # initialize reward and done (will be updated when Phase-1 completes)
        self.phase0_rewards.append(torch.tensor([0.0], device='cpu'))
        self.phase0_dones.append(torch.tensor([False], device='cpu', dtype=torch.bool))
    
    def add_phase1(
        self,
        obs: torch.Tensor,
        anchor: torch.Tensor,
        candidates_actions: torch.Tensor,  # [K]
        candidates_logprobs: torch.Tensor,  # [K]
        candidates_rewards: torch.Tensor,  # [K]
        executed_action: torch.Tensor,
        executed_logprob: torch.Tensor,
        executed_reward: float,
        mask: torch.Tensor,
        done: bool,
    ):
        """Add Phase-1 transition."""
        # detach to avoid double backward
        self.phase1_obs.append(obs.detach().cpu())
        self.phase1_anchors.append(anchor.detach().cpu())
        self.phase1_actions.append(candidates_actions.detach().cpu())
        self.phase1_logprobs.append(candidates_logprobs.detach().cpu())
        self.phase1_rewards.append(candidates_rewards.detach().cpu())
        self.phase1_executed_actions.append(executed_action.detach().cpu())
        self.phase1_executed_logprobs.append(executed_logprob.detach().cpu())
        self.phase1_executed_rewards.append(executed_reward)
        self.phase1_masks.append(mask.detach().cpu())
        self.phase1_dones.append(done)
    
    
    def get_phase0_data(self) -> Dict[str, torch.Tensor]:
        """Get Phase-0 data as tensors."""
        return {
            "obs": torch.stack(self.phase0_obs, dim=0).to(self.device),
            "actions": torch.stack(self.phase0_actions, dim=0).to(self.device),
            "logprobs": torch.stack(self.phase0_logprobs, dim=0).to(self.device),
            "values": torch.stack(self.phase0_values, dim=0).to(self.device),
            "rewards": torch.stack(self.phase0_rewards, dim=0).to(self.device),
            "dones": torch.stack(self.phase0_dones, dim=0).to(self.device),
            "masks": torch.stack(self.phase0_masks, dim=0).to(self.device),
        }
    
    def get_phase1_data(self) -> Dict:
        """Get Phase-1 data."""
        # phase-1 data has variable K and variable mask sizes, so we'll handle it specially
        # pad masks to max size (170) for consistent stacking
        max_mask_size = 170
        padded_masks = []
        for mask in self.phase1_masks:
            mask_size = mask.shape[-1]
            if mask_size < max_mask_size:
                # pad with False (invalid actions)
                padding = torch.zeros(mask.shape[:-1] + (max_mask_size - mask_size,), dtype=torch.bool)
                padded_mask = torch.cat([mask, padding], dim=-1)
            else:
                padded_mask = mask
            padded_masks.append(padded_mask)
        
        return {
            "obs": torch.stack(self.phase1_obs, dim=0).to(self.device),
            "anchors": torch.stack(self.phase1_anchors, dim=0).to(self.device),
            "candidates_actions": self.phase1_actions,  # list of tensors
            "candidates_logprobs": self.phase1_logprobs,
            "candidates_rewards": self.phase1_rewards,
            "executed_actions": torch.stack(self.phase1_executed_actions, dim=0).to(self.device),
            "executed_logprobs": torch.stack(self.phase1_executed_logprobs, dim=0).to(self.device),
            "executed_rewards": torch.tensor(self.phase1_executed_rewards, device=self.device),
            "masks": torch.stack(padded_masks, dim=0).to(self.device),
            "dones": torch.tensor(self.phase1_dones, device=self.device, dtype=torch.bool),
        }
    
    def clear(self):
        """Clear buffer."""
        self.phase0_obs.clear()
        self.phase0_actions.clear()
        self.phase0_logprobs.clear()
        self.phase0_values.clear()
        self.phase0_rewards.clear()
        self.phase0_dones.clear()
        self.phase0_masks.clear()
        self.phase0_env_indices.clear()
        
        self.phase1_obs.clear()
        self.phase1_anchors.clear()
        self.phase1_actions.clear()
        self.phase1_logprobs.clear()
        self.phase1_rewards.clear()
        self.phase1_masks.clear()
        self.phase1_executed_actions.clear()
        self.phase1_executed_logprobs.clear()
        self.phase1_executed_rewards.clear()
        self.phase1_dones.clear()


def visualize_action(
    grid: np.ndarray,
    r1: int,
    c1: int,
    r2: int,
    c2: int,
    turn: int,
    reward: float,
    total_reward: float,
):
    """Visualize the grid with the selected rectangle highlighted."""
    print("\n" + "=" * 70)
    print(f"Turn: {turn} | Reward: {reward:.1f} | Total Reward: {total_reward:.1f}")
    print(f"Action: Rectangle from ({r1},{c1}) to ({r2},{c2})")
    print("=" * 70)
    
    # Extract rectangle values
    rect_values = []
    rect_sum = 0
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            val = grid[r, c]
            rect_values.append(val)
            rect_sum += val
    
    print(f"Rectangle values: {rect_values}")
    print(f"Rectangle sum: {rect_sum}")
    print()
    
    # Print grid with rectangle highlighted
    for r in range(10):
        row_str = []
        for c in range(17):
            val = grid[r, c]
            # Highlight rectangle cells
            if r1 <= r <= r2 and c1 <= c <= c2:
                row_str.append(f"[{val:2d}]")
            else:
                row_str.append(f" {val:2d} ")
        print("".join(row_str))
    
    print("=" * 70 + "\n")


def make_env(seed: int, initial_grid: Optional[np.ndarray] = None, curriculum_updates: int = 400):
    """Create environment."""
    env = Sum10GymEnv(initial_grid=initial_grid)
    env = TwoPhaseWrapper(env, curriculum_legal_only=True, curriculum_updates=curriculum_updates)
    return env


def collect_rollouts(
    envs: List[TwoPhaseWrapper],
    policy: CNNPolicy,
    buffer: RolloutBuffer,
    config: Config,
    frozen_policy: Optional[CNNPolicy] = None,
    visualize: bool = False,
    render_env_idx: int = 0,
    current_update: Optional[int] = None,
):
    """Collect rollouts from environments."""
    if frozen_policy is None:
        frozen_policy = policy
    
    # track actions for visualization
    visualization_data = []
    
    # initial observations
    obs_list = []
    for env in envs:
        obs, _ = env.reset()
        obs_list.append(obs)
    obs = torch.stack(obs_list, dim=0).to(next(policy.parameters()).device)
    
    for step in range(config.rollout_steps):
        # get action masks and phases
        masks_list = []
        phases = []
        for env in envs:
            mask = env.get_action_mask()
            masks_list.append(mask)
            phases.append(env.phase)
        
        # separate Phase-0 and Phase-1 envs
        phase0_indices = [i for i, p in enumerate(phases) if p == 0]
        phase1_indices = [i for i, p in enumerate(phases) if p == 1]
        phase0_mask = torch.tensor([i in phase0_indices for i in range(len(envs))], device=obs.device)
        phase1_mask = ~phase0_mask
        
        # phase-0: select anchor
        if phase0_mask.any():
            phase0_obs = obs[phase0_mask]
            phase0_masks = torch.stack([masks_list[i] for i in phase0_indices], dim=0).to(obs.device)
            phase0_actions, phase0_logprobs,                 phase0_values = policy.get_action_and_value(
                phase0_obs, phase0_masks
            )
            
            # store Phase-0 data
            for i, env_idx in enumerate(phase0_indices):
                buffer.add_phase0(
                    obs[env_idx:env_idx+1],
                    phase0_actions[i:i+1],
                    phase0_logprobs[i:i+1],
                    phase0_values[i:i+1],
                    masks_list[env_idx].unsqueeze(0),
                    env_idx,
                )
        
        # phase-1: select extent with GRPO
        if phase1_mask.any():
            phase1_obs = obs[phase1_mask]
            phase1_masks_list = [masks_list[i] for i in phase1_indices]
            phase1_env_indices = phase1_indices
            
            # get anchors for Phase-1 envs
            phase1_anchors = []
            for env_idx in phase1_env_indices:
                env = envs[env_idx]
                anchor_idx = env.anchor_to_flat_idx(*env.selected_anchor)
                phase1_anchors.append(anchor_idx)
            phase1_anchors = torch.tensor(phase1_anchors, device=obs.device)
            
            # sample K candidates from frozen policy
            all_candidates_actions = []
            all_candidates_logprobs = []
            all_candidates_rewards = []
            
            for i, env_idx in enumerate(phase1_env_indices):
                env = envs[env_idx]
                anchor_idx = phase1_anchors[i].item()
                
                # get valid action mask for this env
                valid_mask = phase1_masks_list[i]
                valid_action_count = valid_mask.sum().item()
                
                # enhanced debug logging for Phase-1 (log every 50 updates)
                if current_update is not None and current_update % 50 == 0:
                    # count legal actions if curriculum is active
                    legal_count = valid_action_count
                    if env.curriculum_legal_only and env.current_update < env.curriculum_updates * 2:
                        legal_mask = env.get_legal_only_mask()
                        legal_count = legal_mask.sum().item() if legal_mask.numel() > 0 else 0
                    
                    wandb.log({
                        "debug/phase1_valid_action_count": valid_action_count,
                        "debug/phase1_legal_count": legal_count,
                    }, commit=False)
                
                # skip if no valid actions (shouldn't happen in normal flow, but handle gracefully)
                if valid_action_count == 0:
                    # use dummy values - this shouldn't happen if curriculum/constraints work correctly
                    all_candidates_actions.append(torch.zeros(config.grpo_k, dtype=torch.long, device=obs.device))
                    all_candidates_logprobs.append(torch.zeros(config.grpo_k, device=obs.device))
                    all_candidates_rewards.append(torch.zeros(config.grpo_k, device=obs.device))
                    continue
                
                # sample K candidates
                # for Phase-1, we need to create a full-size mask (170) with only valid positions
                # Phase-1 action space is variable, so we pad the mask
                full_mask = torch.zeros(170, dtype=torch.bool, device=obs.device)
                # extract only the True values from valid_mask and place them at the start
                # find indices where valid_mask is True
                valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1).to(obs.device)
                # place them at the beginning of full_mask
                full_mask[:valid_action_count] = True
                
                with torch.no_grad():
                    logits, _ = frozen_policy(phase1_obs[i:i+1], full_mask.unsqueeze(0))
                    # extract only valid logits
                    valid_logits = logits[0][:valid_action_count]
                    
                    # apply temperature for diversity (higher temp = more exploration)
                    scaled_logits = valid_logits / max(config.grpo_temperature, 1e-8)
                    
                    dist_scaled = torch.distributions.Categorical(logits=scaled_logits)
                    candidates = dist_scaled.sample((config.grpo_k,))
                    
                    # compute logprobs with original (unscaled) logits for correct probability
                    dist_original = torch.distributions.Categorical(logits=valid_logits)
                    candidates_logprobs = dist_original.log_prob(candidates)
                
                # convert candidate indices back to original mask indices
                # candidates are indices [0, valid_action_count), need to map to actual valid_indices
                candidates_original_indices = valid_indices[candidates]
                
                # simulate each candidate to get rewards
                candidates_rewards = []
                for k in range(config.grpo_k):
                    # use the original index from the valid_mask
                    reward = simulate_action_reward(
                        env.game_env,
                        anchor_idx,
                        candidates_original_indices[k].item(),
                        env,
                        illegal_penalty=config.illegal_penalty,
                    )
                    candidates_rewards.append(reward)
                candidates_rewards = torch.tensor(candidates_rewards, device=obs.device)
                
                # check reward diversity - critical for GRPO
                reward_std = candidates_rewards.std().item()
                reward_range = candidates_rewards.max().item() - candidates_rewards.min().item()
                
                # enhanced debug logging for candidate rewards (log every 50 updates)
                if current_update is not None and current_update % 50 == 0:
                    # compute relative advantages for diversity metric
                    mean_reward = candidates_rewards.mean().item()
                    rel_advantages = candidates_rewards - mean_reward
                    rel_adv_std = rel_advantages.std().item()
                    
                    wandb.log({
                        "debug/phase1_candidate_rewards_mean": candidates_rewards.mean().item(),
                        "debug/phase1_candidate_rewards_std": reward_std,
                        "debug/phase1_candidate_rewards_range": reward_range,
                        "debug/phase1_reward_diversity": rel_adv_std,
                        "debug/phase1_candidate_rewards_min": candidates_rewards.min().item(),
                        "debug/phase1_candidate_rewards_max": candidates_rewards.max().item(),
                        "debug/phase1_num_legal_candidates": (candidates_rewards > 0).sum().item(),
                        "debug/phase1_num_penalized_candidates": (candidates_rewards == config.illegal_penalty).sum().item(),
                    }, commit=False)
                
                # store candidates with original indices for consistency
                all_candidates_actions.append(candidates_original_indices)
                all_candidates_logprobs.append(candidates_logprobs)
                all_candidates_rewards.append(candidates_rewards)
            
            # execute best candidate (or sample from policy)
            executed_actions = []
            executed_logprobs = []
            executed_rewards = []
            
            for i, env_idx in enumerate(phase1_env_indices):
                # use best candidate (highest reward)
                best_idx = torch.argmax(all_candidates_rewards[i])
                executed_action = all_candidates_actions[i][best_idx]
                executed_logprob = all_candidates_logprobs[i][best_idx]
                executed_reward = all_candidates_rewards[i][best_idx].item()
                
                executed_actions.append(executed_action.unsqueeze(0))
                executed_logprobs.append(executed_logprob.unsqueeze(0))
                executed_rewards.append(executed_reward)
            
            # store Phase-1 data
            for i, env_idx in enumerate(phase1_env_indices):
                buffer.add_phase1(
                    obs[env_idx:env_idx+1],
                    phase1_anchors[i:i+1],
                    all_candidates_actions[i],
                    all_candidates_logprobs[i],
                    all_candidates_rewards[i],
                    executed_actions[i],
                    executed_logprobs[i],
                    executed_rewards[i],
                    phase1_masks_list[i].unsqueeze(0),
                    False,  # done will be updated after step
                )
        
        # step environments
        new_obs_list = []
        rewards_list = []
        dones_list = []
        phase0_reward_indices = []
        
        phase0_action_map = {}
        if phase0_mask.any():
            phase0_indices = torch.where(phase0_mask)[0]
            for i, env_idx in enumerate(phase0_indices):
                phase0_action_map[env_idx.item()] = phase0_actions[i].item()
        
        phase1_action_map = {}
        if phase1_mask.any():
            phase1_indices = torch.where(phase1_mask)[0]
            for i, env_idx in enumerate(phase1_indices):
                phase1_action_map[env_idx.item()] = executed_actions[i].item()
        
        for env_idx, env in enumerate(envs):
            if env_idx in phase0_action_map:
                # phase-0: step with anchor action
                action_idx = phase0_action_map[env_idx]
                obs_new, reward, terminated, truncated, info = env.step(action_idx)
                new_obs_list.append(obs_new)
                rewards_list.append(0.0)  # no reward in Phase-0
                dones_list.append(False)
                phase0_reward_indices.append(env_idx)
            elif env_idx in phase1_action_map:
                # phase-1: step with executed extent action
                action_idx = phase1_action_map[env_idx]
                
                # get rectangle coordinates for visualization before step
                if visualize and env_idx == render_env_idx:
                    r1, c1 = env.selected_anchor
                    r2, c2 = env.flat_idx_to_extent(r1, c1, action_idx)
                    grid_before = env.game_env.grid.copy()
                    turn_before = env.game_env.turn
                
                obs_new, reward, terminated, truncated, info = env.step(action_idx)
                
                # store visualization data after step (only for valid moves with reward > 0)
                if visualize and env_idx == render_env_idx and reward > 0:
                    visualization_data.append((
                        reward,
                        grid_before,
                        r1, c1, r2, c2,
                        turn_before,
                    ))
                
                new_obs_list.append(obs_new)
                rewards_list.append(reward)
                dones_list.append(terminated or truncated)
            else:
                # should not happen
                obs_new, reward, terminated, truncated, info = env.reset()
                new_obs_list.append(obs_new)
                rewards_list.append(0.0)
                dones_list.append(False)
        
        obs = torch.stack(new_obs_list, dim=0).to(obs.device)
        rewards = torch.tensor(rewards_list, device=obs.device, dtype=torch.float32)
        dones = torch.tensor(dones_list, device=obs.device, dtype=torch.bool)
        
        # update rewards for Phase-1 completions
        # Phase-1 rewards go to the corresponding Phase-0 transition
        if phase1_mask.any():
            phase1_indices = torch.where(phase1_mask)[0]
            for i, env_idx in enumerate(phase1_indices):
                # find the most recent Phase-0 transition for this env
                for j in range(len(buffer.phase0_env_indices) - 1, -1, -1):
                    if buffer.phase0_env_indices[j] == env_idx.item():
                        # assign Phase-1 reward to this Phase-0 transition
                        buffer.phase0_rewards[j] = torch.tensor([rewards[env_idx].item()], device='cpu')
                        buffer.phase0_dones[j] = torch.tensor([dones[env_idx].item()], device='cpu', dtype=torch.bool)
                        break
        
        # reset done environments
        for env_idx, done in enumerate(dones):
            if done:
                obs_new, _ = envs[env_idx].reset()
                obs[env_idx] = obs_new
    
    return visualization_data


def train(config: Config, use_wandb: bool = True):
    """Main training loop.
    
    Args:
        config: Training configuration
        use_wandb: Whether to use wandb logging (default: True)
    """
    print("Starting training setup...")
    
    # initialize wandb
    if use_wandb:
        # set wandb to use a temp directory to avoid cluttering repo
        import os
        import tempfile
        os.environ["WANDB_DIR"] = tempfile.gettempdir()
        
        wandb.init(
            project="fruit-box-grpo",
            name=f"grpo_seed{config.seed}",
            config={
                "num_envs": config.num_envs,
                "rollout_steps": config.rollout_steps,
                "max_updates": config.max_updates,
                "gamma": config.gamma,
                "gae_lambda": config.gae_lambda,
                "phase0_lr": config.phase0_lr,
                "phase0_clip_eps": config.phase0_clip_eps,
                "phase0_target_kl": config.phase0_target_kl,
                "phase0_value_coef": config.phase0_value_coef,
                "phase1_lr": config.phase1_lr,
                "phase1_clip_eps": config.phase1_clip_eps,
                "entropy_coef": config.entropy_coef,
                "entropy_target": config.entropy_target,
                "entropy_penalty_coef": config.entropy_penalty_coef,
                "epochs": config.epochs,
                "grad_clip": config.grad_clip,
                "grpo_k": config.grpo_k,
                "grpo_tau": config.grpo_tau,
                "grpo_temperature": config.grpo_temperature,
                "frozen_refresh_interval": config.frozen_refresh_interval,
                "curriculum_updates": config.curriculum_updates,
                "illegal_penalty": config.illegal_penalty,
                "batch_size": config.batch_size,
                "lr_warmup_steps": config.lr_warmup_steps,
                "min_reward_std": config.min_reward_std,
                "seed": config.seed,
            },
            tags=["grpo", "fruit-box", "two-phase"],
        )
        print("Wandb initialized!")
    
    # set seeds
    print("Setting seeds...")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # setup
    print("Creating device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # create directories
    print("Creating directories...")
    import os
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # create environments (create first, reset later to avoid segfault)
    print(f"Creating {config.num_envs} environments...")
    envs = []
    for i in range(config.num_envs):
        if (i + 1) % 10 == 0:
            print(f"  Created {i+1}/{config.num_envs} environments...")
        env = make_env(config.seed + i, curriculum_updates=config.curriculum_updates)
        envs.append(env)
    print(f"All {len(envs)} environments created")
    
    # create policy
    print("Creating policy...")
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    print("Policy created")
    
    # create separate optimizers for Phase-0 and Phase-1
    # learning rates will be warmed up during training
    print("Creating optimizers...")
    phase0_optimizer = torch.optim.Adam(policy.parameters(), lr=config.phase0_lr)
    phase1_optimizer = torch.optim.Adam(policy.parameters(), lr=config.phase1_lr)
    print("Optimizers created")
    
    # learning rate warmup helper function
    def get_lr_multiplier(step: int, warmup_steps: int) -> float:
        """Get learning rate multiplier for warmup."""
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0
    
    # create frozen policy for GRPO candidate sampling
    frozen_policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    frozen_policy.load_state_dict(policy.state_dict())
    frozen_policy.eval()
    print("Frozen policy created")
    
    # create buffer
    print("Creating buffer...")
    buffer = RolloutBuffer(config.rollout_steps, config.num_envs, (4, 10, 17), device)
    print("Buffer created")
    
    
    # training loop
    global_step = 0
    for update in tqdm(range(config.max_updates), desc="Training"):
        # learning rate warmup (recommended: 10-20 steps)
        lr_mult = get_lr_multiplier(update, config.lr_warmup_steps)
        if lr_mult < 1.0:
            for param_group in phase0_optimizer.param_groups:
                param_group['lr'] = config.phase0_lr * lr_mult
            for param_group in phase1_optimizer.param_groups:
                param_group['lr'] = config.phase1_lr * lr_mult
        
        # update curriculum
        for env in envs:
            env.set_curriculum_update(update)
        
        # refresh frozen policy periodically
        if update > 0 and update % config.frozen_refresh_interval == 0:
            frozen_policy.load_state_dict(policy.state_dict())
            frozen_policy.eval()
            print(f"Frozen policy refreshed at update {update}")
        
        # collect rollouts
        visualize_this_update = (update % config.render_interval == 0)
        visualization_data = collect_rollouts(
            envs, policy, buffer, config,
            frozen_policy=frozen_policy,
            visualize=visualize_this_update,
            render_env_idx=config.render_env_idx,
            current_update=update,
        )
        
        # visualize actions if requested
        if visualize_this_update and visualization_data:
            print(f"\n{'='*70}")
            print(f"VISUALIZATION - Update {update}")
            print(f"{'='*70}")
            total_reward = 0.0
            for reward, grid, r1, c1, r2, c2, turn in visualization_data:
                total_reward += reward
                visualize_action(grid, r1, c1, r2, c2, turn, reward, total_reward)
        
        # get data
        phase0_data = buffer.get_phase0_data()
        phase1_data = buffer.get_phase1_data()
        
        # compute advantages for Phase-0
        phase0_advantages, phase0_returns = compute_gae(
            phase0_data["rewards"].transpose(0, 1),
            phase0_data["values"].transpose(0, 1),
            phase0_data["dones"].transpose(0, 1),
            config.gamma,
            config.gae_lambda,
        )
        phase0_advantages = phase0_advantages.transpose(0, 1)
        phase0_returns = phase0_returns.transpose(0, 1)
        
        # flatten for training
        phase0_obs_flat = phase0_data["obs"].reshape(-1, *phase0_data["obs"].shape[2:])
        phase0_actions_flat = phase0_data["actions"].reshape(-1)
        phase0_logprobs_flat = phase0_data["logprobs"].reshape(-1)
        phase0_advantages_flat = phase0_advantages.reshape(-1)
        phase0_returns_flat = phase0_returns.reshape(-1)
        phase0_masks_flat = phase0_data["masks"].reshape(-1, phase0_data["masks"].shape[-1])
        
        # normalize returns
        phase0_returns_flat = (phase0_returns_flat - phase0_returns_flat.mean()) / (
            phase0_returns_flat.std() + 1e-8
        )
        
        # scale advantages so std(A) ≈ 1
        adv_std = phase0_advantages_flat.std()
        if adv_std > 1e-8:
            phase0_advantages_flat = phase0_advantages_flat / adv_std
        
        # update Phase-0 (PPO)
        phase0_losses = []
        for epoch in range(config.epochs):
            # shuffle
            indices = torch.randperm(len(phase0_obs_flat), device=device)
            
            for start in range(0, len(phase0_obs_flat), config.batch_size):
                end = start + config.batch_size
                batch_indices = indices[start:end]
                
                batch_obs = phase0_obs_flat[batch_indices]
                batch_actions = phase0_actions_flat[batch_indices]
                batch_old_logprobs = phase0_logprobs_flat[batch_indices]
                batch_advantages = phase0_advantages_flat[batch_indices]
                batch_returns = phase0_returns_flat[batch_indices]
                batch_masks = phase0_masks_flat[batch_indices]
                
                loss, info = compute_ppo_loss(
                    policy,
                    batch_obs,
                    batch_actions,
                    batch_old_logprobs,
                    batch_advantages,
                    batch_returns,
                    batch_masks,
                    config.phase0_clip_eps,
                    config.phase0_value_coef,
                    config.entropy_coef,
                    config.entropy_target,
                    config.entropy_penalty_coef,
                )
                
                # check KL constraint (compute KL divergence)
                # we need to compute new logprobs to check KL
                with torch.no_grad():
                    logits, _ = policy(batch_obs, batch_masks)
                    new_logprobs_batch = []
                    for b in range(batch_obs.size(0)):
                        valid_mask = batch_masks[b]
                        valid_logits = logits[b][valid_mask]
                        dist = torch.distributions.Categorical(logits=valid_logits)
                        new_logprobs_batch.append(dist.log_prob(batch_actions[b]))
                    new_logprobs_batch = torch.stack(new_logprobs_batch)
                    
                    kl_div = (batch_old_logprobs - new_logprobs_batch).mean().item()
                    if kl_div > config.phase0_target_kl:
                        # skip update if KL too large
                        continue
                
                phase0_optimizer.zero_grad()
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
                
                # Log gradient norm if close to clipping threshold (indicates potential instability)
                if use_wandb and total_norm > config.grad_clip * 0.9:
                    wandb.log({"debug/grad_norm": total_norm.item()}, step=update, commit=False)
                
                phase0_optimizer.step()
                
                phase0_losses.append(info)
        
        # update Phase-1 (GRPO)
        phase1_losses = []
        if len(phase1_data["obs"]) > 0:
            # process Phase-1 data (variable K)
            phase1_obs_list = phase1_data["obs"]
            phase1_anchors_list = phase1_data["anchors"]
            phase1_candidates_actions_list = phase1_data["candidates_actions"]
            phase1_candidates_logprobs_list = phase1_data["candidates_logprobs"]
            phase1_candidates_rewards_list = phase1_data["candidates_rewards"]
            phase1_masks_list = phase1_data["masks"]
            
            # batch process (handle variable K)
            for epoch in range(config.epochs):
                # create batches
                num_phase1_samples = len(phase1_obs_list)
                indices = torch.randperm(num_phase1_samples, device=device)
                
                for start in range(0, num_phase1_samples, config.batch_size):
                    end = min(start + config.batch_size, num_phase1_samples)
                    batch_indices = indices[start:end]
                    
                    # gather batch (will filter out dummy entries next)
                    batch_candidates_actions = [phase1_candidates_actions_list[i] for i in batch_indices]
                    batch_candidates_logprobs = [phase1_candidates_logprobs_list[i] for i in batch_indices]
                    batch_candidates_rewards = [phase1_candidates_rewards_list[i] for i in batch_indices]
                    
                    # filter out entries with no valid actions (dummy entries)
                    # these have all-zero actions/rewards from the continue case
                    valid_batch_indices = []
                    filtered_candidates_actions = []
                    filtered_candidates_logprobs = []
                    filtered_candidates_rewards = []
                    filtered_batch_obs = []
                    filtered_batch_anchors = []
                    filtered_batch_masks = []
                    
                    for i, idx in enumerate(batch_indices):
                        # check mask to see if there are valid actions (more reliable than checking actions/rewards)
                        mask = phase1_masks_list[idx]
                        valid_count = mask.sum().item() if mask.numel() > 0 else 0
                        
                        # skip if no valid actions (dummy entry from continue case)
                        if valid_count == 0:
                            continue
                        
                        valid_batch_indices.append(i)
                        filtered_candidates_actions.append(batch_candidates_actions[i])
                        filtered_candidates_logprobs.append(batch_candidates_logprobs[i])
                        filtered_candidates_rewards.append(batch_candidates_rewards[i])
                        filtered_batch_obs.append(phase1_obs_list[idx])
                        filtered_batch_anchors.append(phase1_anchors_list[idx])
                        filtered_batch_masks.append(mask)
                    
                    # skip if all entries in batch were filtered out
                    if len(filtered_candidates_actions) == 0:
                        continue
                    
                    # stack candidates (pad to max K if needed)
                    max_k = max(len(a) for a in filtered_candidates_actions)
                    batch_size = len(filtered_candidates_actions)
                    
                    padded_actions = torch.zeros(batch_size, max_k, dtype=torch.long, device=device)
                    padded_logprobs = torch.zeros(batch_size, max_k, device=device)
                    padded_rewards = torch.zeros(batch_size, max_k, device=device)
                    
                    for i, (actions, logprobs, rewards) in enumerate(zip(
                        filtered_candidates_actions,
                        filtered_candidates_logprobs,
                        filtered_candidates_rewards,
                    )):
                        k = len(actions)
                        padded_actions[i, :k] = actions
                        padded_logprobs[i, :k] = logprobs
                        padded_rewards[i, :k] = rewards
                    
                    # stack filtered data
                    batch_obs = torch.cat(filtered_batch_obs, dim=0)
                    batch_anchors = torch.cat(filtered_batch_anchors, dim=0)
                    batch_masks = torch.cat(filtered_batch_masks, dim=0)
                    
                    # compute GRPO loss
                    loss, info = compute_grpo_loss(
                        policy,
                        batch_obs,
                        batch_anchors.squeeze(-1),
                        padded_actions,
                        padded_logprobs,
                        padded_rewards,
                        batch_masks,
                        config.phase1_clip_eps,
                    )
                    
                    phase1_optimizer.zero_grad()
                    loss.backward()
                    total_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
                    
                    # Log gradient norm if close to clipping threshold (indicates potential instability)
                    if use_wandb and total_norm > config.grad_clip * 0.9:
                        wandb.log({"debug/phase1_grad_norm": total_norm.item()}, step=update, commit=False)
                    
                    phase1_optimizer.step()
                    
                    phase1_losses.append(info)
        
        # logging
        if phase0_losses:
            # convert all tensor values to scalars before averaging
            # skip tensors that aren't scalar metrics (like 'new_logprobs')
            phase0_losses_scalar = []
            for d in phase0_losses:
                scalar_dict = {}
                for k, v in d.items():
                    if k == 'new_logprobs':
                        # skip this - it's a tensor used only for KL check
                        continue
                    if isinstance(v, torch.Tensor):
                        if v.numel() == 1:
                            scalar_dict[k] = v.item()
                        else:
                            # skip multi-element tensors
                            continue
                    else:
                        scalar_dict[k] = v
                phase0_losses_scalar.append(scalar_dict)
            
            if phase0_losses_scalar and len(phase0_losses_scalar) > 0:
                avg_phase0_loss = {k: np.mean([d[k] for d in phase0_losses_scalar]) for k in phase0_losses_scalar[0]}
                print(f"Update {update}: Phase-0 loss: {avg_phase0_loss.get('ppo_loss', 0):.4f}")
                
                # Monitor entropy and intervene before collapse
                current_entropy = avg_phase0_loss.get('entropy', 5.0)
                
                # Early warning: entropy dropping too fast - intervene proactively
                if current_entropy < 0.3 and update > 100:
                    print(f"\n{'='*70}")
                    print(f"WARNING: Entropy dangerously low ({current_entropy:.3f}) at update {update}")
                    print(f"Triggering early intervention: increasing entropy_coef and reducing learning rates")
                    print(f"{'='*70}\n")
                    
                    # Increase entropy coefficient immediately
                    config.entropy_coef = min(0.15, config.entropy_coef * 1.5)
                    
                    # Reduce learning rates to stabilize
                    for param_group in phase0_optimizer.param_groups:
                        param_group['lr'] *= 0.7
                    for param_group in phase1_optimizer.param_groups:
                        param_group['lr'] *= 0.7
                    
                    # Log intervention
                    if use_wandb:
                        wandb.log({
                            "recovery/entropy_rescue": 1.0,
                            "recovery/new_entropy_coef": config.entropy_coef,
                            "recovery/new_phase0_lr": phase0_optimizer.param_groups[0]['lr'],
                            "recovery/new_phase1_lr": phase1_optimizer.param_groups[0]['lr'],
                        }, step=update)
            
            # log to wandb
            if use_wandb:
                wandb.log({
                    "update": update,
                    "phase0/ppo_loss": avg_phase0_loss.get('ppo_loss', 0),
                    "phase0/policy_loss": avg_phase0_loss.get('policy_loss', 0),
                    "phase0/value_loss": avg_phase0_loss.get('value_loss', 0),
                    "phase0/entropy": avg_phase0_loss.get('entropy', 0),
                    "phase0/entropy_penalty": avg_phase0_loss.get('entropy_penalty', 0),
                    "phase0/clip_fraction": avg_phase0_loss.get('clip_fraction', 0),
                }, step=update)
        
        if phase1_losses:
            # convert all tensor values to scalars before averaging
            phase1_losses_scalar = []
            for d in phase1_losses:
                scalar_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
                phase1_losses_scalar.append(scalar_dict)
            
            avg_phase1_loss = {k: np.mean([d[k] for d in phase1_losses_scalar]) for k in phase1_losses_scalar[0]}
            print(f"Update {update}: Phase-1 loss: {avg_phase1_loss.get('grpo_loss', 0):.4f}")
            
            # log to wandb
            if use_wandb:
                wandb.log({
                    "update": update,
                    "phase1/grpo_loss": avg_phase1_loss.get('grpo_loss', 0),
                    "phase1/mean_advantage": avg_phase1_loss.get('mean_advantage', 0),
                    "phase1/mean_ratio": avg_phase1_loss.get('mean_ratio', 0),
                    "phase1/clip_fraction": avg_phase1_loss.get('clip_fraction', 0),
                    "phase1/reward_diversity_std": avg_phase1_loss.get('reward_diversity_std', 0),
                    "phase1/reward_range": avg_phase1_loss.get('reward_range', 0),
                    "phase1/relative_advantage_std": avg_phase1_loss.get('relative_advantage_std', 0),
                }, step=update)
        
        # log rollout statistics
        if use_wandb and phase0_data:
            # compute statistics from rollouts
            total_rewards = phase0_data["rewards"].sum().item()
            mean_reward = phase0_data["rewards"].mean().item()
            valid_moves = (phase0_data["rewards"] > 0).sum().item()
            total_moves = phase0_data["rewards"].numel()
            
            wandb.log({
                "rollout/total_reward": total_rewards,
                "rollout/mean_reward": mean_reward,
                "rollout/valid_moves": valid_moves,
                "rollout/total_moves": total_moves,
                "rollout/legality_rate": valid_moves / max(total_moves, 1),
            }, step=update)
        
        # clear buffer
        buffer.clear()
        global_step += config.rollout_steps * config.num_envs
        
        # checkpoint
        if (update + 1) % config.checkpoint_interval == 0:
            torch.save(policy.state_dict(), f"{config.checkpoint_dir}/policy_{update+1}.pt")
    
    # save final checkpoint
    print(f"\nSaving final checkpoint...")
    torch.save(policy.state_dict(), f"{config.checkpoint_dir}/policy_final.pt")
    print(f"Training complete! Final checkpoint saved to {config.checkpoint_dir}/policy_final.pt")
    
    # finalize wandb run
    if use_wandb:
        wandb.finish()
        print("Wandb run completed!")
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()
    
    config = Config(seed=args.seed)
    train(config, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()