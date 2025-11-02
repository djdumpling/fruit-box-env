"""PPO (Proximal Policy Optimization) utilities."""
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional


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
    
    # Convert dones to float for arithmetic
    dones_float = dones.float()
    
    # Compute TD errors
    # values[:, :-1] predicts values[:, 1:], with last value being bootstrap
    # We need to handle the last step specially
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    # Process backwards through sequence
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            # Last step: no next value
            next_value = 0.0
        else:
            next_value = values[:, t + 1]
        
        # TD error
        delta = rewards[:, t] + gamma * next_value * (1 - dones_float[:, t]) - values[:, t]
        
        # GAE
        advantages[:, t] = last_gae = delta + gamma * lam * (1 - dones_float[:, t]) * last_gae
    
    # Returns = advantages + values
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
    value_clip: Optional[float] = None,
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
    
    Returns:
        loss: Scalar loss tensor
        info: Dictionary with loss components and statistics
    """
    # Get current policy outputs
    logits, values = policy(obs, action_mask)  # [batch_size, action_dim], [batch_size, 1]
    values = values.squeeze(-1)  # [batch_size]
    
    # Compute new logprobs
    # Extract valid logits for each batch item
    new_logprobs = []
    entropies = []
    for b in range(obs.size(0)):
        valid_mask = action_mask[b]  # [action_dim]
        valid_logits = logits[b][valid_mask]  # [valid_action_count]
        
        dist = torch.distributions.Categorical(logits=valid_logits)
        # actions[b] is index into valid action space
        new_logprobs.append(dist.log_prob(actions[b]))
        entropies.append(dist.entropy())
    
    new_logprobs = torch.stack(new_logprobs)  # [batch_size]
    entropies = torch.stack(entropies)  # [batch_size]
    
    # Policy loss (PPO clipped)
    ratio = torch.exp(new_logprobs - old_logprobs)  # [batch_size]
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss1 = ratio * advantages
    policy_loss2 = clipped_ratio * advantages
    policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
    
    # Value loss (MSE, optionally clipped)
    # NOTE: value_clip now clips the absolute error, not the squared error
    # This prevents extreme value updates while still allowing learning
    if value_clip is not None and value_clip > 0:
        value_error = returns - values
        # Clip absolute error, then square for MSE
        value_error_clipped = torch.clamp(value_error, -value_clip, value_clip)
        value_loss = (value_error_clipped ** 2).mean()
    else:
        value_loss = ((values - returns) ** 2).mean()
    
    # Entropy bonus
    entropy_bonus = entropies.mean()
    
    # Total loss
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
    
    # Statistics
    info = {
        "ppo_loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy_bonus.item(),
        "mean_advantage": advantages.mean().item(),
        "mean_ratio": ratio.mean().item(),
        "clip_fraction": ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float().mean().item(),
    }
    
    return loss, info

