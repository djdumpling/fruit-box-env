"""PPO (Proximal Policy Optimization) utilities."""
import torch
import torch.nn as nn
from typing import Tuple, Dict


def map_action_to_valid_space(action: int, valid_mask: torch.Tensor) -> int:
    """Map an action index from full action space to valid action space.
    
    Args:
        action: Action index in full action space (0-169)
        valid_mask: Boolean mask indicating valid actions [action_dim]
    
    Returns:
        Index in valid action space (0 to valid_action_count-1)
    
    Raises:
        ValueError: If action is invalid and no valid actions exist
    """
    # Get indices where mask is True
    valid_indices = torch.nonzero(valid_mask, as_tuple=False)  # [valid_action_count, 1]
    if valid_indices.dim() > 1:
        valid_indices = valid_indices.squeeze(-1)  # [valid_action_count]
    
    # Handle case where valid_indices might be 0-d or empty
    if valid_indices.numel() == 0:
        raise ValueError(f"No valid actions exist in mask")
    
    # Check if action is valid
    if action >= valid_mask.size(0) or not valid_mask[action]:
        # Action is invalid - this shouldn't happen, but handle gracefully
        return 0  # Use first valid action as fallback
    
    # Find the index in valid_indices that matches action
    # valid_indices is a 1D tensor, so we can directly compare
    matches = (valid_indices == action).nonzero(as_tuple=False)
    if matches.numel() == 0:
        # Action was valid when sampled but mask changed - use first valid action
        return 0
    else:
        # matches is a 1D tensor with indices into valid_indices
        return matches.item() if matches.dim() == 0 else matches[0].item()


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.995,
    lam: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: [batch_size, seq_len] reward tensor
        values: [batch_size, seq_len] value estimates
        dones: [batch_size, seq_len] done flags
        gamma: Discount factor
        lam: GAE lambda parameter
    
    Returns:
        advantages: [batch_size, seq_len] advantage estimates
        returns: [batch_size, seq_len] return estimates
    """
    batch_size, seq_len = rewards.shape
    device = rewards.device
    
    # convert dones to float for arithmetic
    dones_float = dones.float()
    
    # compute TD errors
    # values[:, :-1] predicts values[:, 1:], with last value being bootstrap
    # we need to handle the last step specially
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    # process backwards through sequence
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            # last step: no next value
            next_value = 0.0
        else:
            next_value = values[:, t + 1]
        
        # TD error
        delta = rewards[:, t] + gamma * next_value * (1 - dones_float[:, t]) - values[:, t]
        
        # GAE
        advantages[:, t] = last_gae = delta + gamma * lam * (1 - dones_float[:, t]) * last_gae
    
    # returns = advantages + values
    returns = advantages + values
    
    return advantages, returns


def compute_ppo_loss(
    policy: nn.Module,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    action_mask: torch.Tensor,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    entropy_target: float = 0.0,
    entropy_penalty_coef: float = 0.0
) -> Tuple[torch.Tensor, Dict]:
    """Compute PPO clipped loss.
    
    Args:
        policy: Policy network
        obs: [batch_size, 4, 10, 17] observations
        actions: [batch_size] action indices
        old_logprobs: [batch_size] log probabilities from old policy
        advantages: [batch_size] advantage estimates
        returns: [batch_size] return estimates
        action_mask: [batch_size, action_dim] action masks
        clip_eps: PPO clipping epsilon
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        entropy_target: Target minimum entropy (default: 0.0 = disabled)
        entropy_penalty_coef: Penalty coefficient for entropy below target
    
    Returns:
        loss: Scalar loss tensor
        info: Dictionary with loss components and statistics
    """
    # get current policy outputs
    logits, values = policy(obs, action_mask)  # [batch_size, action_dim], [batch_size, 1]
    values = values.squeeze(-1)  # [batch_size]
    
    # compute new logprobs
    # extract valid logits for each batch item
    new_logprobs = []
    entropies = []
    for b in range(obs.size(0)):
        valid_mask = action_mask[b]  # [action_dim]
        valid_logits = logits[b][valid_mask]  # [valid_action_count]
        
        # Map action from full action space to valid action space
        # actions[b] is an index into the full action space (0-169)
        action_idx = actions[b].item()
        mapped_action_idx = map_action_to_valid_space(action_idx, valid_mask)
        
        dist = torch.distributions.Categorical(logits=valid_logits)
        # mapped_action_idx is now index into valid action space
        new_logprobs.append(dist.log_prob(torch.tensor(mapped_action_idx, device=actions.device)))
        entropies.append(dist.entropy())
    
    new_logprobs = torch.stack(new_logprobs)  # [batch_size]
    entropies = torch.stack(entropies)  # [batch_size]
    
    # policy loss (PPO clipped)
    ratio = torch.exp(new_logprobs - old_logprobs)  # [batch_size]
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss1 = ratio * advantages
    policy_loss2 = clipped_ratio * advantages
    policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
    
    # value loss (MSE)
    value_loss = ((values - returns) ** 2).mean()
    
    # entropy bonus
    entropy_bonus = entropies.mean()
    
    # entropy floor penalty (to prevent over-confidence)
    entropy_penalty = 0.0
    if entropy_target > 0.0 and entropy_penalty_coef > 0.0:
        entropy_shortfall = torch.clamp(entropy_target - entropy_bonus, min=0.0)
        entropy_penalty = entropy_penalty_coef * entropy_shortfall
    
    # total loss
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus + entropy_penalty
    
    # statistics
    info = {
        "ppo_loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy_bonus.item(),
        "entropy_penalty": entropy_penalty.item() if isinstance(entropy_penalty, torch.Tensor) else entropy_penalty,
        "mean_advantage": advantages.mean().item(),
        "mean_ratio": ratio.mean().item(),
        "clip_fraction": ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float().mean().item(),
        # note: new_logprobs is not included as it's a tensor used only for KL computation
    }
    
    return loss, info

