""" python rl/train/train_grpo.py --seed 42 --load-checkpoint artifacts/policy_sft_epoch80.pt """

import sys
from pathlib import Path
# add project root to path for imports (go up 2 levels from rl/train/train_grpo.py)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import wandb

from rl.envs.sum10_env import Sum10GymEnv
from rl.envs.split_wrapper import TwoPhaseWrapper
from rl.models.policy import CNNPolicy
from rl.algo.ppo import compute_gae, compute_ppo_loss, map_action_to_valid_space
from rl.algo.grpo import compute_grpo_loss, simulate_action_reward


def is_wandb_artifact(path: str) -> bool:
    return "/" in path and ":" in path and not Path(path).exists()


def load_checkpoint_from_wandb(artifact_path: str) -> str:
    print(f"Downloading wandb artifact: {artifact_path}")
    # Initialize wandb run to access artifacts
    # Note: This creates a temporary run just for artifact access
    run = wandb.init()
    try:
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
    finally:
        wandb.finish()


@dataclass
class Config:
    """Training config for GRPO strategy learning from SFT policy
    
    SFT policy learned legality (0.9998 negative accuracy, 0.94 accuracy) but uses random strategy.
    We'll learn optimal strategy (minimal area approach) with aggressive exploration that gradually reduces.
    """
    # data collection
    num_envs: int = 16
    rollout_steps: int = 128
    batch_size: int = 512
    epochs: int = 4
    
    # phase-0 (PPO) hyperparameters
    phase0_lr: float = 1e-5  # lower (policy is good)
    phase0_clip_eps: float = 0.06  # was 0.12, tighter to avoid breaking the good policy
    phase0_target_kl: float = 0.003  # was 0.008, keep updates small
    phase0_value_coef: float = 0.8
    
    # phase-1 (GRPO) hyperparameters
    phase1_lr: float = 2e-5  # lower (policy is good)
    phase1_clip_eps: float = 0.12  # was 0.2
    phase1_target_ratio: float = 1.8  # was 2.5
    grpo_k: int = 24 # higher chance of finding minimal rewards
    frozen_refresh_interval: int = 30  # reduced from 100 to prevent frozen policy from becoming too outdated
    grpo_temperature: float = 1.8 # (more diverse sampling)
    min_reward_std: float = 0.01
    
    # shared hyperparameters
    max_updates: int = 2500  # more updates for strategy learning
    gamma: float = 0.998  # standard discount, no augment factor
    gae_lambda: float = 0.95
    entropy_coef: float = 0.03  # start higher for exploration
    entropy_target: float = 0.3  # start higher
    entropy_penalty_coef: float = 0.2  # stronger penalty
    grad_clip: float = 1.0
    lr_warmup_steps: int = 30  # was 50, policy is already good so less warmup needed
    
    # exploration schedule
    exploration_schedule: str = "linear"  # linear decay of entropy over time
    exploration_start_coef: float = 0.05  # starting entropy coefficient
    exploration_end_coef: float = 0.02  # ending entropy coefficient
    exploration_start_target: float = 0.5  # starting entropy target
    exploration_end_target: float = 0.3  # ending entropy target
    
    # curriculum learning
    curriculum_updates: int = 0  # disabled, SFT already handles legal moves
    illegal_penalty: float = -0.1
    
    # minimal area curriculum
    use_minimal_area_curriculum: bool = False  # use grids from minimal_area policy
    minimal_area_dataset: str = "djdumpling/fruit-box-minimal-area"
    minimal_area_num_grids: int = 100  # number of grids to load
    
    # minimal-area strategy hyperparameters
    area_penalty_coef: float = 0.3  # penalty coefficient for rewards > 2 in early turns
    early_turn_threshold: int = 30  # turns where area penalty applies
    early_penalty_weight: float = 1.0  # multiplier for early-turn penalty
    
    # other
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 500
    render_interval: int = 5
    render_env_idx: int = 0
    load_checkpoint: Optional[str] = None  # path to checkpoint to load at start
    use_legal_only_masks: bool = False  # use legal-only masks (only needed for epoch 0 SFT checkpoints; epoch 1+ learned legality with all geometric masks)


class RolloutBuffer:
    """Buffer for storing rollout data"""
    
    def __init__(self, rollout_steps: int, num_envs: int, obs_shape: Tuple[int, ...], device: str):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.device = device
        
        # phase-0 data (anchor selection)
        self.phase0_obs = []
        self.phase0_actions = []
        self.phase0_logprobs = []
        self.phase0_values = []
        self.phase0_rewards = []
        self.phase0_dones = []
        self.phase0_masks = []
        self.phase0_env_indices = []  # track which env each transition belongs to
        self.phase0_valid = []  # track validity of moves (True if legal, False if illegal)
        
        # phase-1 data (extent selection)
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
        """Add Phase-0 transition"""
        # detach to avoid double backward
        self.phase0_obs.append(obs.detach().cpu())
        self.phase0_actions.append(action.detach().cpu())
        self.phase0_logprobs.append(logprob.detach().cpu())
        self.phase0_values.append(value.detach().cpu())
        self.phase0_masks.append(mask.detach().cpu())
        self.phase0_env_indices.append(env_idx)
        # initialize reward, done, and valid (will be updated when Phase-1 completes)
        self.phase0_rewards.append(torch.tensor([0.0], device='cpu'))
        self.phase0_dones.append(torch.tensor([False], device='cpu', dtype=torch.bool))
        self.phase0_valid.append(torch.tensor([False], device='cpu', dtype=torch.bool))
    
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
        """Add Phase-1 transition"""
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
        """Get Phase-0 data as tensors"""
        return {
            "obs": torch.stack(self.phase0_obs, dim=0).to(self.device),
            "actions": torch.stack(self.phase0_actions, dim=0).to(self.device),
            "logprobs": torch.stack(self.phase0_logprobs, dim=0).to(self.device),
            "values": torch.stack(self.phase0_values, dim=0).to(self.device),
            "rewards": torch.stack(self.phase0_rewards, dim=0).to(self.device),
            "dones": torch.stack(self.phase0_dones, dim=0).to(self.device),
            "masks": torch.stack(self.phase0_masks, dim=0).to(self.device),
            "valid": torch.stack(self.phase0_valid, dim=0).to(self.device),
        }
    
    def get_phase1_data(self) -> Dict:
        """Get Phase-1 data"""
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
        """Clear buffer"""
        self.phase0_obs.clear()
        self.phase0_actions.clear()
        self.phase0_logprobs.clear()
        self.phase0_values.clear()
        self.phase0_rewards.clear()
        self.phase0_dones.clear()
        self.phase0_masks.clear()
        self.phase0_env_indices.clear()
        self.phase0_valid.clear()
        
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
    """Visualize the grid with the selected rectangle highlighted"""
    # extract rectangle values
    rect_values = []
    rect_sum = 0
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            val = grid[r, c]
            rect_values.append(val)
            rect_sum += val
    
    print(f"Turn {turn}: ({r1},{c1})→({r2},{c2}) | Reward: {reward:.1f} | Total: {total_reward:.1f} | Sum: {rect_sum}")
    
    # print grid with rectangle highlighted
    for r in range(10):
        row_str = []
        for c in range(17):
            val = grid[r, c]
            # highlight rectangle cells
            if r1 <= r <= r2 and c1 <= c <= c2:
                row_str.append(f"[{val:2d}]")
            else:
                row_str.append(f" {val:2d} ")
        print("".join(row_str))
    print()


def get_exploration_coef(update: int, max_updates: int, start_coef: float, end_coef: float) -> float:
    """Linear schedule from start_coef to end_coef over max_updates"""
    if max_updates <= 0:
        return start_coef
    progress = min(update / max_updates, 1.0)
    return start_coef + (end_coef - start_coef) * progress


def load_minimal_area_grids(dataset_name: str, num_grids: int) -> List[np.ndarray]:
    """Load initial grids from dataset filtered by agent_tag='minimal_area'"""
    from datasets import load_dataset
    hf_dataset = load_dataset(dataset_name, split="train")
    grids = []
    seen_episodes = set()
    for row in hf_dataset:
        if row.get("agent_tag") == "minimal_area":
            ep_id = row.get("episode_id")
            if ep_id is not None and ep_id not in seen_episodes and row.get("step", -1) == 0:
                grid_data = row.get("grid")
                if grid_data is not None:
                    grids.append(np.array(grid_data, dtype=np.uint8))
                    seen_episodes.add(ep_id)
                    if len(grids) >= num_grids:
                        break
    if len(grids) == 0:
        print(f"WARNING: No minimal_area grids found in dataset {dataset_name}. Using random grids instead.")
    else:
        print(f"Loaded {len(grids)} initial grids from minimal_area policy")
    return grids


def make_env(seed: int, initial_grid: Optional[np.ndarray] = None, curriculum_updates: int = 400, initial_grids: Optional[List[np.ndarray]] = None, grid_index: int = 0, curriculum_legal_only: bool = False):
    """Create environment"""
    # if initial_grids provided, use the grid at grid_index (cycling through)
    if initial_grids is not None and len(initial_grids) > 0:
        grid_to_use = initial_grids[grid_index % len(initial_grids)]
        env = Sum10GymEnv(initial_grid=grid_to_use, seed=seed)
    elif initial_grid is not None:
        env = Sum10GymEnv(initial_grid=initial_grid, seed=seed)
    else:
        env = Sum10GymEnv(seed=seed)
    env = TwoPhaseWrapper(env, curriculum_legal_only=curriculum_legal_only, curriculum_updates=curriculum_updates)
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
    """Collect rollouts from environments"""
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
            if config.use_legal_only_masks:
                # use legal-only masks (required when loading from SFT)
                if env.phase == 0:
                    # phase-0: compute legal anchors (anchors with at least one legal extent)
                    legal_anchors_set = set()
                    grid = env.game_env.grid
                    for anchor_r1 in range(10):
                        for anchor_c1 in range(17):
                            anchor_idx = env.anchor_to_flat_idx(anchor_r1, anchor_c1)
                            # check if this anchor has any legal extents
                            max_valid_count = (10 - anchor_r1) * (17 - anchor_c1)
                            has_legal = False
                            for extent_idx in range(max_valid_count):
                                r2_test, c2_test = env.flat_idx_to_extent(anchor_r1, anchor_c1, extent_idx)
                                if env.game_env.box_sum(anchor_r1, anchor_c1, r2_test, c2_test) == 10:
                                    reward_test = env.game_env.box_nonzero_count(anchor_r1, anchor_c1, r2_test, c2_test)
                                    if reward_test > 0:
                                        has_legal = True
                                        break
                            if has_legal:
                                legal_anchors_set.add(anchor_idx)
                    # build mask: True only at legal anchor positions
                    mask = torch.zeros(170, dtype=torch.bool)
                    for legal_anchor_idx in sorted(legal_anchors_set):
                        mask[legal_anchor_idx] = True
                else:
                    # phase-1: use legal-only mask (only extents that sum to 10)
                    # get_legal_only_mask() returns compact mask, we'll handle padding later
                    mask = env.get_legal_only_mask()
            else:
                # use standard action mask (all geometrically valid actions)
                # policy learned legality in SFT, so it can handle all actions and avoid illegal ones
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
            # pad phase-0 masks to 170 if needed
            phase0_masks_padded = []
            for i in phase0_indices:
                mask = masks_list[i]
                if mask.shape[0] < 170:
                    padded = torch.zeros(170, dtype=torch.bool)
                    padded[:mask.shape[0]] = mask
                    phase0_masks_padded.append(padded)
                else:
                    phase0_masks_padded.append(mask)
            phase0_masks = torch.stack(phase0_masks_padded, dim=0).to(obs.device)
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
                valid_mask_compact = phase1_masks_list[i]
                # handle variable-size masks (legal-only masks may be smaller than 170)
                # map compact mask to full 170-space
                r1, c1 = env.selected_anchor
                action_dim = (10 - r1) * (17 - c1)
                valid_mask = torch.zeros(170, dtype=torch.bool, device=obs.device)
                if valid_mask_compact.shape[0] <= action_dim:
                    # map compact indices to full space
                    for compact_idx in range(min(valid_mask_compact.shape[0], action_dim)):
                        if valid_mask_compact[compact_idx]:
                            # extent index in compact space maps to same index in full space for this anchor
                            valid_mask[compact_idx] = True
                else:
                    # mask is already in full space
                    valid_mask[:valid_mask_compact.shape[0]] = valid_mask_compact[:170]
                valid_action_count = valid_mask.sum().item()
                
                # CRITICAL FIX: When not using legal-only masks, we need to filter to only legal actions
                # The policy learned legality, but stochastic sampling can still pick illegal actions
                # So we filter the mask to only include legal actions before sampling
                if not config.use_legal_only_masks:
                    # Get legal-only mask to filter out illegal actions
                    legal_mask_compact = env.get_legal_only_mask()
                    # Map legal mask to full 170-space (same logic as valid_mask)
                    legal_mask_full = torch.zeros(170, dtype=torch.bool, device=obs.device)
                    if legal_mask_compact.shape[0] <= action_dim:
                        for compact_idx in range(min(legal_mask_compact.shape[0], action_dim)):
                            if legal_mask_compact[compact_idx]:
                                legal_mask_full[compact_idx] = True
                    else:
                        legal_mask_full[:legal_mask_compact.shape[0]] = legal_mask_compact[:170]
                    
                    # Intersect valid_mask with legal_mask_full to get only legal actions
                    valid_mask = valid_mask & legal_mask_full
                    valid_action_count = valid_mask.sum().item()
                
                # enhanced debug logging for Phase-1 (log every 50 updates)
                if current_update is not None and current_update % 50 == 0:
                    # count legal actions
                    legal_mask_compact = env.get_legal_only_mask()
                    legal_mask_full = torch.zeros(170, dtype=torch.bool, device=obs.device)
                    if legal_mask_compact.shape[0] <= action_dim:
                        for compact_idx in range(min(legal_mask_compact.shape[0], action_dim)):
                            if legal_mask_compact[compact_idx]:
                                legal_mask_full[compact_idx] = True
                    else:
                        legal_mask_full[:legal_mask_compact.shape[0]] = legal_mask_compact[:170]
                    legal_count = legal_mask_full.sum().item()
                    
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
                # find indices where valid_mask is True (now filtered to only legal actions)
                valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1).to(obs.device)
                # ensure valid_indices is 1D (squeeze might make it 0D if empty, but we already checked valid_action_count > 0)
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)
                # set full_mask to True at the actual valid indices (not necessarily contiguous from 0)
                full_mask[valid_indices] = True
                
                with torch.no_grad():
                    logits, _ = frozen_policy(phase1_obs[i:i+1], full_mask.unsqueeze(0))
                    # extract logits at valid indices (not necessarily at the beginning)
                    valid_logits = logits[0][valid_indices]
                    
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
                # Clear debug info before collecting rewards for this batch
                if hasattr(simulate_action_reward, '_debug_info'):
                    simulate_action_reward._debug_info.clear()
                
                candidates_rewards = []
                for k in range(config.grpo_k):
                    # use the original index from the valid_mask
                    reward = simulate_action_reward(
                        env.game_env,
                        anchor_idx,
                        candidates_original_indices[k].item(),
                        env,
                        illegal_penalty=config.illegal_penalty,
                        area_penalty_coef=config.area_penalty_coef,
                        turn_number=env.game_env.turn,
                        early_turn_threshold=config.early_turn_threshold,
                        early_penalty_weight=config.early_penalty_weight,
                    )
                    candidates_rewards.append(reward)
                candidates_rewards = torch.tensor(candidates_rewards, device=obs.device)
                
                # Collect and log debug info for delayed gratification penalty
                if hasattr(simulate_action_reward, '_debug_info') and len(simulate_action_reward._debug_info) > 0:
                    debug_info = simulate_action_reward._debug_info
                    if current_update is not None and current_update % 50 == 0:
                        # Aggregate debug stats
                        base_rewards = [d['base_reward'] for d in debug_info]
                        turn_numbers = [d['turn_number'] for d in debug_info]
                        time_weights = [d['time_weight'] for d in debug_info]
                        reward_penalties = [d['reward_penalty'] for d in debug_info]
                        final_rewards = [d['final_reward'] for d in debug_info]
                        penalties_applied = sum(1 for d in debug_info if d['penalty_applied'])
                        
                        step_info_rewards = [d['step_info_reward'] for d in debug_info]
                        
                        wandb.log({
                            "debug/reward_calc_step_info_reward_mean": np.mean(step_info_rewards) if step_info_rewards else 0,
                            "debug/reward_penalty_base_reward_mean": np.mean(base_rewards) if base_rewards else 0,
                            "debug/reward_penalty_turn_number_mean": np.mean(turn_numbers) if turn_numbers else 0,
                            "debug/reward_penalty_time_weight_mean": np.mean(time_weights) if time_weights else 0,
                            "debug/reward_penalty_penalty_mean": np.mean(reward_penalties) if reward_penalties else 0,
                            "debug/reward_penalty_final_reward_mean": np.mean(final_rewards) if final_rewards else 0,
                            "debug/reward_penalty_applied_ratio": penalties_applied / len(debug_info) if debug_info else 0,
                            "debug/reward_calc_final_reward_std": np.std(final_rewards) if final_rewards else 0,
                        }, commit=False)
                
                # check reward diversity - critical for GRPO
                reward_std = candidates_rewards.std().item()
                
                # enhanced debug logging for candidate rewards (log every 50 updates)
                if current_update is not None and current_update % 50 == 0:
                    wandb.log({
                        "debug/phase1_candidate_rewards_mean": candidates_rewards.mean().item(),
                        "debug/phase1_candidate_rewards_std": reward_std,
                        "debug/phase1_candidate_rewards_min": candidates_rewards.min().item(),
                        "debug/phase1_candidate_rewards_max": candidates_rewards.max().item(),
                        "debug/phase1_num_legal_candidates": (candidates_rewards >= 0).sum().item(),  # legal = non-negative (includes reward == 0)
                    }, commit=False)
                
                # Debug logging: check if all candidates are the same action
                if current_update is not None and current_update % 50 == 0:
                    unique_actions = torch.unique(candidates_original_indices)
                    unique_action_count = len(unique_actions)
                    wandb.log({
                        "debug/phase1_unique_candidate_actions": unique_action_count,
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
        valid_list = []  # track validity of moves
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
                valid_list.append(True)  # Phase-0 is always valid (just selecting anchor)
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
                
                # track validity from environment info
                is_valid = info.get("valid", True)
                valid_list.append(is_valid)
                
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
                valid_list.append(False)  # reset is not a valid move
        
        obs = torch.stack(new_obs_list, dim=0).to(obs.device)
        rewards = torch.tensor(rewards_list, device=obs.device, dtype=torch.float32)
        dones = torch.tensor(dones_list, device=obs.device, dtype=torch.bool)
        
        # update rewards and validity for Phase-1 completions
        # CRITICAL: Phase-1 completions happen when we step envs that were in Phase-1 at the START of this step
        # These envs had Phase-0 transitions added in the PREVIOUS step
        # So we need to match Phase-1 completions (from envs that were Phase-1) to Phase-0 transitions
        # The key insight: envs that are Phase-1 NOW were Phase-0 in the PREVIOUS step
        # But wait - actually, Phase-0 and Phase-1 transitions are added in the SAME step for different envs
        # Then we step all envs. So:
        # - Env A: Phase-0 → add Phase-0 transition → step → becomes Phase-1
        # - Env B: Phase-1 → add Phase-1 transition → step → becomes Phase-0 or resets
        # So Phase-0 transitions added this step will be completed next step when those envs become Phase-1
        # But Phase-1 transitions added this step complete immediately when we step
        # So we need to match Phase-1 completions (from stepping Phase-1 envs) to Phase-0 transitions from PREVIOUS step
        # Actually wait - let me reconsider. The phases at the START of the step determine what we do:
        # - If Phase-0: we add Phase-0 transition, then step → becomes Phase-1
        # - If Phase-1: we add Phase-1 transition, then step → completes the move, becomes Phase-0
        # So Phase-1 transitions added this step complete when we step (same step)
        # But Phase-0 transitions added this step will complete next step
        # So we need to match Phase-1 completions to Phase-0 transitions from PREVIOUS step
        # But we're updating in the same step, so we need to match to Phase-0 transitions that were added
        # when these envs were Phase-0, which was the PREVIOUS step
        
        # Actually, the simplest approach: match Phase-1 completions (from stepping Phase-1 envs) 
        # to the most recent Phase-0 transition for each env
        if phase1_mask.any():
            phase1_indices = torch.where(phase1_mask)[0]
            for i, env_idx_tensor in enumerate(phase1_indices):
                env_idx = env_idx_tensor.item()
                # Find the most recent Phase-0 transition for this env
                # This should be from the previous step when this env was Phase-0
                found = False
                for j in range(len(buffer.phase0_env_indices) - 1, -1, -1):
                    if buffer.phase0_env_indices[j] == env_idx:
                        # This Phase-0 transition corresponds to this Phase-1 completion
                        # CRITICAL: valid_list is indexed by env_idx (0 to num_envs-1)
                        if env_idx < len(valid_list):
                            is_valid = valid_list[env_idx]
                            buffer.phase0_rewards[j] = torch.tensor([rewards[env_idx].item()], device='cpu')
                            buffer.phase0_dones[j] = torch.tensor([dones[env_idx].item()], device='cpu', dtype=torch.bool)
                            buffer.phase0_valid[j] = torch.tensor([is_valid], device='cpu', dtype=torch.bool)
                            found = True
                        else:
                            print(f"WARNING: env_idx {env_idx} >= len(valid_list) {len(valid_list)}")
                            buffer.phase0_valid[j] = torch.tensor([False], device='cpu', dtype=torch.bool)
                        break
                if not found and step > 0:  # Allow first step to not have matches
                    # This can happen if env resets or if Phase-0 transition wasn't added
                    pass  # Phase-0 transition stays with initial values (reward=0, valid=False)
        
        # reset done environments
        for env_idx, done in enumerate(dones):
            if done:
                obs_new, _ = envs[env_idx].reset()
                obs[env_idx] = obs_new
    
    return visualization_data


def train(config: Config, use_wandb: bool = True):
    """Main training loop
    
    Args:
        config: Training configuration
        use_wandb: Whether to use wandb logging (default: True)
    """
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
                "phase1_target_ratio": config.phase1_target_ratio,
                "entropy_coef": config.entropy_coef,
                "entropy_target": config.entropy_target,
                "entropy_penalty_coef": config.entropy_penalty_coef,
                "epochs": config.epochs,
                "grad_clip": config.grad_clip,
                "grpo_k": config.grpo_k,
                "grpo_temperature": config.grpo_temperature,
                "frozen_refresh_interval": config.frozen_refresh_interval,
                "curriculum_updates": config.curriculum_updates,
                "illegal_penalty": config.illegal_penalty,
                "batch_size": config.batch_size,
                "lr_warmup_steps": config.lr_warmup_steps,
                "min_reward_std": config.min_reward_std,
                "area_penalty_coef": config.area_penalty_coef,
                "early_turn_threshold": config.early_turn_threshold,
                "early_penalty_weight": config.early_penalty_weight,
                "seed": config.seed,
            },
            tags=["grpo", "fruit-box", "two-phase", "minimal-area"],
        )
        print("Wandb initialized!")
    
    # set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {config.seed} | Envs: {config.num_envs}")
    
    # create directories
    import os
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # load minimal_area grids if curriculum enabled
    initial_grids = None
    if config.use_minimal_area_curriculum:
        print(f"Loading minimal_area grids from {config.minimal_area_dataset}...")
        initial_grids = load_minimal_area_grids(config.minimal_area_dataset, config.minimal_area_num_grids)
    
    # create environments (create first, reset later to avoid segfault)
    envs = []
    for i in range(config.num_envs):
        env = make_env(
            config.seed + i, 
            curriculum_updates=config.curriculum_updates, 
            initial_grids=initial_grids, 
            grid_index=i,
            curriculum_legal_only=config.use_legal_only_masks
        )
        envs.append(env)
    print(f"All {len(envs)} environments created")
    print(f"Using legal-only masks: {config.use_legal_only_masks}")
    
    # create policy
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    
    # load checkpoint if provided
    if config.load_checkpoint:
        # Check if checkpoint is a wandb artifact
        checkpoint_path = config.load_checkpoint
        if is_wandb_artifact(checkpoint_path):
            checkpoint_path = load_checkpoint_from_wandb(checkpoint_path)
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # load checkpoint (includes both policy and value heads)
        policy.load_state_dict(checkpoint)
        
        # re-initialize value head: SFT value estimates are wrong for RL returns
        # keep policy head weights from SFT, but reset value head
        # use smaller initial weights to prevent high initial grad norms
        print("Re-initializing value head (SFT value estimates don't match RL returns)...")
        for param in policy.value_head.parameters():
            if len(param.shape) >= 2:
                # use smaller scale for xavier initialization to prevent high initial gradients
                torch.nn.init.xavier_uniform_(param, gain=0.5)  # reduced from default gain=1.0
            else:
                torch.nn.init.zeros_(param)
        
        print("Checkpoint loaded successfully! (Policy head from SFT, value head re-initialized)")
        
        # Verify checkpoint loaded correctly by checking a few weights
        with torch.no_grad():
            sample_weight = policy.phase0_head.weight[0, 0].item()
            print(f"  Verification: phase0_head weight sample = {sample_weight:.6f}")
    else:
        print("Policy created (no checkpoint loaded)")
    
    # Set policy to eval mode for rollout collection (matches test_sft.py behavior)
    # This ensures consistent behavior with how the checkpoint was evaluated
    policy.eval()
    print("Policy set to eval mode for rollout collection")
    
    # Quick sanity check: test policy on a single environment to verify it works
    if config.load_checkpoint:
        print("Running sanity check: testing loaded policy on a single environment...")
        test_env = make_env(
            config.seed + 999,  # use a different seed for test
            curriculum_updates=config.curriculum_updates,
            curriculum_legal_only=config.use_legal_only_masks
        )
        test_obs, _ = test_env.reset()
        test_obs = test_obs.unsqueeze(0).to(device)
        
        # Test Phase-0
        test_mask = test_env.get_action_mask()
        if test_mask.shape[0] < 170:
            padded = torch.zeros(170, dtype=torch.bool)
            padded[:test_mask.shape[0]] = test_mask
            test_mask = padded
        test_mask = test_mask.unsqueeze(0).to(device)
        
        with torch.no_grad():
            test_logits, _ = policy(test_obs, test_mask)
            test_action = test_logits.argmax(dim=1).item()
            print(f"  Sanity check: Phase-0 action selected: {test_action}")
        
        # Step Phase-0
        test_obs, _, _, _, _ = test_env.step(test_action)
        test_obs = test_obs.unsqueeze(0).to(device)
        
        # Test Phase-1
        test_mask = test_env.get_action_mask()
        if test_mask.shape[0] < 170:
            padded = torch.zeros(170, dtype=torch.bool)
            padded[:test_mask.shape[0]] = test_mask
            test_mask = padded
        test_mask = test_mask.unsqueeze(0).to(device)
        
        with torch.no_grad():
            test_logits, _ = policy(test_obs, test_mask)
            test_action = test_logits.argmax(dim=1).item()
            print(f"  Sanity check: Phase-1 action selected: {test_action}")
        
        # Step Phase-1 and check validity
        test_obs, test_reward, terminated, truncated, info = test_env.step(test_action)
        is_valid = info.get("valid", True)
        print(f"  Sanity check: Move valid: {is_valid}, Reward: {test_reward}")
        if not is_valid:
            print(f"  WARNING: Sanity check move was INVALID! This suggests a problem with the loaded policy.")
        else:
            print(f"  Sanity check passed: policy selected a valid move.")
    
    # create separate optimizers for Phase-0 and Phase-1
    # learning rates will be warmed up during training
    print("Creating optimizers...")
    phase0_optimizer = torch.optim.Adam(policy.parameters(), lr=config.phase0_lr)
    phase1_optimizer = torch.optim.Adam(policy.parameters(), lr=config.phase1_lr)
    
    # learning rate warmup helper function
    def get_lr_multiplier(step: int, warmup_steps: int) -> float:
        """Get learning rate multiplier for warmup"""
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0
    
    # create frozen policy for GRPO candidate sampling
    frozen_policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    frozen_policy.load_state_dict(policy.state_dict())
    frozen_policy.eval()
    print("Frozen policy initialized (matches current policy)")
    
    # create buffer
    print("Creating buffer...")
    buffer = RolloutBuffer(config.rollout_steps, config.num_envs, (4, 10, 17), device)
    
    # recovery mechanism: track if we need to override schedule
    entropy_override_coef = None
    entropy_override_target = None
    
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
        
        # update exploration schedule (linear decay)
        # use override if recovery mechanism activated, otherwise use schedule
        if entropy_override_coef is not None:
            current_entropy_coef = entropy_override_coef
            current_entropy_target = entropy_override_target
        else:
            current_entropy_coef = get_exploration_coef(
                update, config.max_updates,
                config.exploration_start_coef, config.exploration_end_coef
            )
            current_entropy_target = get_exploration_coef(
                update, config.max_updates,
                config.exploration_start_target, config.exploration_end_target
            )
        
        # update curriculum
        for env in envs:
            env.set_curriculum_update(update)
        
        # refresh frozen policy periodically
        if update > 0 and update % config.frozen_refresh_interval == 0:
            frozen_policy.load_state_dict(policy.state_dict())
            frozen_policy.eval()
        
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
            print(f"\nUpdate {update} visualization:")
            total_reward = 0.0
            for reward, grid, r1, c1, r2, c2, turn in visualization_data:
                total_reward += reward
                visualize_action(grid, r1, c1, r2, c2, turn, reward, total_reward)
        
        # Set policy to train mode for updates (needed for dropout/batch norm if present)
        policy.train()
        
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
                    current_entropy_coef,
                    current_entropy_target,
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
                        # Map action from full action space to valid action space
                        action_idx = batch_actions[b].item()
                        mapped_action_idx = map_action_to_valid_space(action_idx, valid_mask)
                        dist = torch.distributions.Categorical(logits=valid_logits)
                        new_logprobs_batch.append(dist.log_prob(torch.tensor(mapped_action_idx, device=batch_actions.device)))
                    new_logprobs_batch = torch.stack(new_logprobs_batch)
                    
                    kl_div = (batch_old_logprobs - new_logprobs_batch).mean().item()
                    if kl_div > config.phase0_target_kl:
                        # skip update if KL too large
                        continue
                
                phase0_optimizer.zero_grad()
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
                
                # log gradient norm if close to clipping threshold (indicates potential instability)
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
                    
                    # check mean ratio constraint for Phase-1 (prevent catastrophic policy changes)
                    mean_ratio = info.get('mean_ratio', 1.0)
                    if mean_ratio > config.phase1_target_ratio:
                        # skip update if mean ratio too large (policy changing too fast)
                        if use_wandb:
                            wandb.log({
                                "debug/phase1_ratio_skip": 1.0,
                                "debug/phase1_mean_ratio": mean_ratio,
                            }, step=update, commit=False)
                        continue
                    
                    phase1_optimizer.zero_grad()
                    loss.backward()
                    total_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
                    
                    # log gradient norm if close to clipping threshold (indicates potential instability)
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
                
                # Clear override if entropy has recovered (above threshold)
                if entropy_override_coef is not None and current_entropy > 0.5:
                    print(f"Entropy recovered to {current_entropy:.3f}, clearing override and resuming schedule")
                    entropy_override_coef = None
                    entropy_override_target = None
                
                # Early warning: entropy dropping too fast - intervene proactively
                # Only trigger if override is not already active (to avoid repeated interventions)
                if current_entropy < 0.3 and update > 100 and entropy_override_coef is None:
                    print(f"\nWARNING: Low entropy ({current_entropy:.3f}) at update {update} - adjusting hyperparameters\n")
                    
                    # Override schedule with higher entropy values
                    # Use scheduled values (not override) to calculate new override
                    scheduled_coef = get_exploration_coef(
                        update, config.max_updates,
                        config.exploration_start_coef, config.exploration_end_coef
                    )
                    scheduled_target = get_exploration_coef(
                        update, config.max_updates,
                        config.exploration_start_target, config.exploration_end_target
                    )
                    entropy_override_coef = min(0.15, scheduled_coef * 1.5)
                    entropy_override_target = min(0.5, scheduled_target * 1.2)
                    
                    # Reduce learning rates to stabilize
                    for param_group in phase0_optimizer.param_groups:
                        param_group['lr'] *= 0.7
                    for param_group in phase1_optimizer.param_groups:
                        param_group['lr'] *= 0.7
                    
                    # Log intervention
                    if use_wandb:
                        wandb.log({
                            "recovery/entropy_rescue": 1.0,
                            "recovery/new_entropy_coef": entropy_override_coef,
                            "recovery/new_entropy_target": entropy_override_target,
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
                    "phase0/entropy_excess_penalty": avg_phase0_loss.get('entropy_excess_penalty', 0),
                    "phase0/clip_fraction": avg_phase0_loss.get('clip_fraction', 0),
                    "exploration/entropy_coef": current_entropy_coef,
                    "exploration/entropy_target": current_entropy_target,
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
                    # debug metrics for mean_ratio investigation
                    "debug/phase1_old_logprobs_mean": avg_phase1_loss.get('debug/old_logprobs_mean', 0),
                    "debug/phase1_new_logprobs_mean": avg_phase1_loss.get('debug/new_logprobs_mean', 0),
                    "debug/phase1_ratio_mean": avg_phase1_loss.get('debug/ratio_mean', 0),  # unclamped
                    "debug/phase1_ratio_mean_clamped": avg_phase1_loss.get('debug/ratio_mean_clamped', 0),  # clamped
                    "debug/phase1_ratio_max": avg_phase1_loss.get('debug/ratio_max', 0),
                }, step=update)
        
        # log rollout statistics
        if use_wandb and phase0_data:
            # compute statistics from rollouts
            total_rewards = phase0_data["rewards"].sum().item()
            mean_reward = phase0_data["rewards"].mean().item()
            valid_moves = phase0_data["valid"].sum().item()  # use actual validity instead of rewards > 0
            total_moves = phase0_data["valid"].numel()
            
            # Debug: check if rewards > 0 matches validity
            rewards_positive = (phase0_data["rewards"] > 0).sum().item()
            validity_rate = valid_moves / max(total_moves, 1)
            reward_positive_rate = rewards_positive / max(total_moves, 1)
            
            # Debug: check how many Phase-0 transitions never got updated
            # (should be 0 if all Phase-0 transitions have corresponding Phase-1 completions)
            never_updated = (phase0_data["rewards"] == 0.0).sum().item()
            
            wandb.log({
                "rollout/total_reward": total_rewards,
                "rollout/mean_reward": mean_reward,
                "rollout/total_moves": total_moves,
                "rollout/legality_rate": validity_rate,
                "debug/rollout_rewards_positive": rewards_positive,
                "debug/rollout_reward_positive_rate": reward_positive_rate,
                "debug/rollout_never_updated_count": never_updated,
            }, step=update)
            
            # Print warning if legality rate is suspiciously low
            if validity_rate < 0.5 and update % 10 == 0:
                print(f"WARNING: Low legality rate {validity_rate:.2%} at update {update}")
                print(f"  Valid moves: {valid_moves}/{total_moves}")
                print(f"  Rewards > 0: {rewards_positive}/{total_moves}")
                print(f"  Never updated: {never_updated}/{total_moves}")
        
        # Set policy back to eval mode for next rollout collection
        policy.eval()
        
        # clear buffer
        buffer.clear()
        global_step += config.rollout_steps * config.num_envs
        
        # checkpoint
        if (update + 1) % config.checkpoint_interval == 0:
            checkpoint_path = f"{config.checkpoint_dir}/policy_{update+1}.pt"
            torch.save(policy.state_dict(), checkpoint_path)
            
            # upload checkpoint to wandb as artifact
            if use_wandb:
                artifact = wandb.Artifact(
                    name=f"checkpoint-{update+1}",
                    type="model",
                    description=f"Policy checkpoint at update {update+1}",
                )
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)
    
    # save final checkpoint
    final_checkpoint_path = f"{config.checkpoint_dir}/policy_final.pt"
    torch.save(policy.state_dict(), final_checkpoint_path)
    print(f"\nTraining complete! Final checkpoint: {final_checkpoint_path}")
    
    # upload final checkpoint to wandb as artifact
    if use_wandb:
        artifact = wandb.Artifact(
            name="checkpoint-final",
            type="model",
            description=f"Final policy checkpoint after {config.max_updates} updates",
        )
        artifact.add_file(final_checkpoint_path)
        wandb.log_artifact(artifact)
        wandb.finish()    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--load-checkpoint", type=str, default=None, 
                       help="Path to checkpoint file or wandb artifact (e.g., 'checkpoints/policy.pt' or 'entity/project/artifact-name:v0')")
    parser.add_argument("--use-legal-only-masks", action="store_true",
                       help="Use legal-only masks (only needed for epoch 0 SFT checkpoints; epoch 1+ learned legality)")
    parser.add_argument("--no-legal-only-masks", action="store_true",
                       help="Disable legal-only masks even when loading checkpoints")
    args = parser.parse_args()
    
    # use legal-only masks only if explicitly requested
    use_legal_only = args.use_legal_only_masks
    if args.load_checkpoint and not use_legal_only and not args.no_legal_only_masks:
        # Note: SFT checkpoints from epoch 1+ were trained with all geometric masks (including illegal actions)
        # and learned to avoid illegal actions via set-based losses. They should work fine without legal-only masks.
        # Only epoch 0 SFT checkpoints used legal-only masks (curriculum phase).
        print(f"INFO: Loading SFT checkpoint without --use-legal-only-masks. "
              f"SFT checkpoints from epoch 1+ were trained with all geometric masks and learned legality. "
              f"This should work fine. If you see legality issues, the checkpoint might be from epoch 0.")
    
    config = Config(seed=args.seed, load_checkpoint=args.load_checkpoint, use_legal_only_masks=use_legal_only)
    train(config, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()