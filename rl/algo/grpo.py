"""GRPO (Group Relative Policy Optimization) loss computation."""
import torch
import torch.nn as nn
from typing import Tuple, Dict
import numpy as np

from fruit_box import Sum10Env


def compute_grpo_loss(
    policy: nn.Module,
    obs: torch.Tensor,
    anchor: torch.Tensor,  # [batch_size] anchor indices
    actions: torch.Tensor,  # [batch_size, K] sampled extent indices
    old_logprobs: torch.Tensor,  # [batch_size, K]
    rewards: torch.Tensor,  # [batch_size, K] immediate rewards
    action_mask: torch.Tensor,  # [batch_size, action_dim]
    clip_eps: float = 0.2
) -> Tuple[torch.Tensor, Dict]:
    """Compute GRPO loss for Phase-1 actions.
    
    Args:
        policy: Policy network
        obs: [batch_size, 4, 10, 17] observations
        anchor: [batch_size] anchor indices (flat indices)
        actions: [batch_size, K] sampled extent indices
        old_logprobs: [batch_size, K] log probabilities from frozen policy
        rewards: [batch_size, K] immediate rewards from simulation
        action_mask: [batch_size, action_dim] action masks
        clip_eps: PPO clipping epsilon
    
    Returns:
        loss: Scalar loss tensor
        info: Dictionary with loss components and statistics
    """
    batch_size, K = actions.shape
    
    # compute relative advantages: A_k = R_k - mean(R)
    mean_reward = rewards.mean(dim=1, keepdim=True)  # [batch_size, 1]
    advantages = rewards - mean_reward  # [batch_size, K]
    
    # get current policy logits (already masked)
    # detach to avoid double backward through Phase-0 updates
    with torch.no_grad():
        logits_detached, _ = policy(obs, action_mask)  # [batch_size, action_dim]
    
    # for training, we need gradients, so compute again without detaching
    logits, _ = policy(obs, action_mask)  # [batch_size, action_dim]
    
    # compute logprobs for sampled actions
    # for each batch item, extract valid logits and compute logprobs
    new_logprobs = []
    for b in range(batch_size):
        # get valid action mask for this batch item
        # NOTE: this mask comes from get_phase1_data() which pads sparse masks
        # the mask may have True values at sparse positions (e.g., index 42)
        # but during collection, we created full_mask with True at [0, valid_count)
        # we need to normalize: create a mask with True at first valid_count positions
        valid_mask = action_mask[b]  # [action_dim] - may be padded with sparse True values
        valid_count = valid_mask.sum().item()
        
        # skip if no valid actions (shouldn't happen, but handle gracefully)
        if valid_count == 0:
            # return zero logprobs for dummy actions
            batch_logprobs = torch.zeros(actions.shape[1], device=actions.device)
            new_logprobs.append(batch_logprobs)
            continue
        
        # get actions for this batch item
        # NOTE: actions[b] contains original indices from the sparse mask space
        # we need to map them back to compact indices [0, valid_count)
        batch_actions_original = actions[b]  # [K] - original indices
        
        # get valid indices in original space (where True values are in the padded mask)
        valid_indices_original = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1).to(actions.device)
        
        # map original actions to compact indices [0, valid_count)
        # create a mapping tensor: mapping[orig_idx] = compact_idx
        # this maps sparse original indices to compact indices [0, valid_count)
        max_orig_idx = valid_indices_original.max().item() if valid_indices_original.numel() > 0 else 0
        if max_orig_idx >= 0:
            mapping = torch.full((max_orig_idx + 1,), -1, dtype=torch.long, device=actions.device)
            mapping[valid_indices_original] = torch.arange(valid_indices_original.numel(), device=actions.device)
            
            # map batch actions to compact indices
            clamped_actions = batch_actions_original.clamp(0, max_orig_idx)
            batch_actions_compact = mapping[clamped_actions]
            
            # handle any invalid mappings (shouldn't happen, but clamp to valid range)
            batch_actions_compact = torch.clamp(batch_actions_compact, 0, valid_count - 1)
        else:
            # edge case: no valid indices
            batch_actions_compact = torch.zeros_like(batch_actions_original)
        
        # create normalized mask for policy: True at first valid_count positions
        # this matches the full_mask format used during collection
        normalized_mask = torch.zeros(valid_mask.shape[0], dtype=torch.bool, device=valid_mask.device)
        normalized_mask[:valid_count] = True
        
        # re-compute logits with normalized mask to match collection format
        # this ensures logits[b][:valid_count] corresponds to compact indices [0, valid_count)
        obs_b = obs[b:b+1]
        normalized_mask_b = normalized_mask.unsqueeze(0)
        logits_normalized, _ = policy(obs_b, normalized_mask_b)
        logits_normalized = logits_normalized[0]  # [action_dim]
        
        # extract valid logits (first valid_count positions)
        valid_logits = logits_normalized[:valid_count]  # [valid_action_count]
        
        # compute logprobs over valid actions using compact indices
        dist = torch.distributions.Categorical(logits=valid_logits)
        batch_logprobs = dist.log_prob(batch_actions_compact)
        new_logprobs.append(batch_logprobs)
    
    new_logprobs = torch.stack(new_logprobs, dim=0)  # [batch_size, K]
    
    # compute importance weights
    ratio = torch.exp(new_logprobs - old_logprobs)  # [batch_size, K]
    
    # PPO clipped loss
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    loss1 = ratio * advantages
    loss2 = clipped_ratio * advantages
    loss = -torch.min(loss1, loss2).mean()
    
    # statistics
    reward_std = rewards.std().item()
    reward_range = rewards.max().item() - rewards.min().item()
    rel_adv_std = advantages.std().item()
    
    # count unique rewards (another diversity metric)
    unique_rewards = len(torch.unique(rewards))
    
    info = {
        "grpo_loss": loss.item(),
        "mean_advantage": advantages.mean().item(),
        "mean_ratio": ratio.mean().item(),
        "clip_fraction": ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float().mean().item(),
        "reward_diversity_std": reward_std,  # standard deviation of rewards within group
        "reward_range": reward_range,  # range of rewards (max - min)
        "relative_advantage_std": rel_adv_std,  # std of relative advantages (key diversity metric)
        "unique_reward_count": unique_rewards,  # number of distinct reward values
    }
    
    return loss, info


def simulate_action_reward(
    env: Sum10Env,
    anchor_idx: int,
    extent_idx: int,
    wrapper,
    illegal_penalty: float = -0.05,
    legal_action_bonus: float = 0.0,
) -> float:
    """Simulate action and return immediate reward.
    
    Args:
        env: Sum10Env instance (will be cloned)
        anchor_idx: Flat index for anchor (r1, c1)
        extent_idx: Flat index for extent (r2, c2) given anchor
        wrapper: TwoPhaseWrapper instance (for conversion functions)
        illegal_penalty: Penalty for illegal moves (default: -0.05)
        legal_action_bonus: Bonus for legal moves (default: 0.0)
    
    Returns:
        reward: Immediate reward from the action (penalty applied if illegal, bonus if legal)
    """
    # clone environment state
    cloned_env = Sum10Env()
    cloned_env.grid = env.grid.copy()
    cloned_env.rebuild_prefix_sums()
    
    # convert indices to coordinates
    r1, c1 = wrapper.flat_idx_to_anchor(anchor_idx)
    r2, c2 = wrapper.flat_idx_to_extent(r1, c1, extent_idx)
    
    # execute action
    step_info = cloned_env.step(r1, c1, r2, c2)
    
    if step_info.valid:
        return float(step_info.reward) + legal_action_bonus
    else:
        # apply penalty for illegal moves (consistent with actual execution)
        return illegal_penalty

