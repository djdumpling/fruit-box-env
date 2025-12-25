"""Loss computation for SFT training."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from fruit_box import Sum10Env
from rl.train.sft_utils import flat_idx_to_extent


def compute_sft_loss(
    policy: nn.Module,
    obs: torch.Tensor,
    actions: torch.Tensor,
    masks: torch.Tensor,
    is_positive: Optional[torch.Tensor] = None,
    negative_loss_weight: float = 2.0,
    legal_actions_sets: Optional[List[set]] = None,
    illegal_mass_alpha: float = 2.0,
    illegal_mass_beta: float = 3.0,
    topk_illegal_k: int = 10,
    topk_illegal_delta: float = 5.0,
    legal_mass_bonus_zeta: float = 0.5,
    use_set_based_losses: bool = True,
    rewards: Optional[torch.Tensor] = None,  # [batch_size] reward for each example
    grid_densities: Optional[torch.Tensor] = None,  # [batch_size] grid density for each example
    step_nums: Optional[torch.Tensor] = None,  # [batch_size] step number for each example
    use_context_aware_reward_weighting: bool = True,
    context_aware_early_threshold: float = 0.5,
    context_aware_trajectory_threshold: int = 20,
    sum_prediction_loss_weight: float = 0.1,  # weight for MSE loss on sum predictions
    phase0_loss_weight: float = 1.0,  # weight multiplier for Phase-0 losses
    phase1_loss_weight: float = 1.5,  # weight multiplier for Phase-1 losses
    phase1_set_based_multiplier: float = 1.5,  # multiplier for set-based losses in Phase-1
) -> Tuple[torch.Tensor, Dict]:
    """Compute SFT loss with set-based legality losses
    
    For positive examples: standard cross-entropy to maximize probability of correct (legal) action
    For negative examples: penalize high probability on illegal action using -log(1 - prob(illegal))
    
    Set-based losses (when use_set_based_losses=True):
    - Illegal mass loss: penalize sum of probabilities on ALL illegal actions
    - Top-K illegal loss: penalize top-K illegal actions by probability
    - Legal mass bonus: reward high probability on legal actions
    
    Sum prediction loss:
    - MSE loss between predicted and actual rectangle sums (only for Phase-1 examples)
    """
    logits, value, sum_predictions = policy(obs, masks)  # [batch_size, 170] for logits and sum_predictions
    
    # Extract grid from observation (Channel 0: normalized values * 9.0)
    # Extract phase from observation (Channel 3)
    # Extract anchor position from observation (Channel 2) for Phase-1
    grids = (obs[:, 0, :, :] * 9.0).cpu().numpy().astype(np.uint8)  # [batch_size, 10, 17]
    phases = obs[:, 3, 0, 0].cpu().numpy()  # [batch_size] - 0.0 for Phase-0, 1.0 for Phase-1
    
    # Compute actual rectangle sums for Phase-1 examples
    sum_prediction_losses = []
    sum_prediction_errors = []
    temp_env = Sum10Env()
    
    for b in range(obs.size(0)):
        if phases[b] > 0.5:  # Phase-1 (extent selection)
            # Extract anchor position from Channel 2
            anchor_mask = obs[b, 2, :, :].cpu().numpy()  # [10, 17]
            anchor_pos = np.argwhere(anchor_mask > 0.5)
            if len(anchor_pos) == 0:
                continue  # No anchor selected, skip sum prediction loss
            r1, c1 = int(anchor_pos[0][0]), int(anchor_pos[0][1])
            
            # Get grid state
            grid = grids[b]
            temp_env.reset(grid=grid.copy())
            
            # Compute actual sums for all valid extent candidates
            mask = masks[b].cpu().numpy()  # [170]
            valid_indices = np.where(mask)[0]
            
            actual_sums = np.zeros(170, dtype=np.float32)
            for extent_idx in valid_indices:
                r2, c2 = flat_idx_to_extent(r1, c1, extent_idx)
                # Safety check: ensure extent is within bounds
                if 0 <= r2 < 10 and 0 <= c2 < 17 and r1 <= r2 and c1 <= c2:
                    actual_sum = temp_env.box_sum(r1, c1, r2, c2)
                    actual_sums[extent_idx] = float(actual_sum)
                # If out of bounds, actual_sum remains 0 (invalid extent)
            
            # Compute MSE loss for sum predictions (only on valid actions)
            valid_sum_predictions = sum_predictions[b][valid_indices]  # [valid_count]
            valid_actual_sums = torch.from_numpy(actual_sums[valid_indices]).to(sum_predictions.device)  # [valid_count]
            
            if len(valid_indices) > 0:
                mse_loss = F.mse_loss(valid_sum_predictions, valid_actual_sums)
                sum_prediction_losses.append(mse_loss)
                
                # Track mean absolute error for logging
                mae = torch.mean(torch.abs(valid_sum_predictions - valid_actual_sums))
                sum_prediction_errors.append(mae.item())
    
    # Aggregate sum prediction loss
    if sum_prediction_losses:
        sum_pred_loss = torch.stack(sum_prediction_losses).mean()
    else:
        # No Phase-1 examples in this batch - create zero tensor that can be part of computation graph
        sum_pred_loss = torch.tensor(0.0, device=obs.device, requires_grad=True)
    mean_sum_error = np.mean(sum_prediction_errors) if sum_prediction_errors else 0.0
    
    # compute loss for each sample
    losses = []
    set_based_losses = []  # illegal mass, top-k, legal bonus
    total = 0  # positive example count
    negative_correct = 0
    negative_total = 0
    legal_prediction_count = 0
    total_prediction_count = 0
    
    # Phase-specific legal accuracy tracking (PRIMARY metric)
    # We track whether predicted actions are legal, not whether they match expert exactly
    phase0_total = 0
    phase0_legal_correct = 0  # predicted anchor has at least one legal extent
    phase1_total = 0
    phase1_legal_correct = 0  # predicted extent is legal (sum=10)
    
    # Entropy tracking
    entropies = []
    
    # metrics for set-based losses
    illegal_mass_sum = 0.0
    topk_illegal_sum = 0.0
    legal_mass_sum = 0.0
    set_based_count = 0
    # Phase-specific metrics
    phase0_losses = []
    phase1_losses = []
    phase1_illegal_mass_sum = 0.0
    phase1_set_based_count = 0
    
    for b in range(obs.size(0)):
        mask = masks[b]  # [170]
        valid_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)  # [valid_count]
        # ensure valid_indices is 1D
        if valid_indices.dim() == 0:
            valid_indices = valid_indices.unsqueeze(0)
        valid_count = valid_indices.numel()
        
        if valid_count == 0:
            continue
        
        # extract valid logits (only at positions where mask is True)
        valid_logits = logits[b][valid_indices]  # [valid_action_count]
        action = actions[b].item()
        
        # map action index to position in valid_indices
        action_pos = (valid_indices == action).nonzero(as_tuple=False)
        if action_pos.numel() == 0:
            continue
        if action_pos.numel() > 1:
            action_compact = action_pos[0].item()
        else:
            action_compact = action_pos.squeeze().item()
        
        # check if this is a negative example
        is_neg = is_positive is not None and not is_positive[b].item() if is_positive is not None else False
        
        # compute probabilities over valid actions
        probs = F.softmax(valid_logits, dim=0)  # [valid_count]
        
        # compute entropy (for exploration/confidence tracking)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        entropies.append(entropy.item())
        
        # determine phase
        is_phase1 = phases[b] > 0.5
        
        legal_actions_set = None
        if legal_actions_sets is not None and b < len(legal_actions_sets):
            legal_actions_set = legal_actions_sets[b]

        # compute set-based losses if enabled and we have legal actions info
        if use_set_based_losses and legal_actions_set is not None:
            # convert valid_indices to set for fast lookup
            valid_indices_set = set(valid_indices.cpu().numpy().tolist())
            
            # identify legal vs illegal actions in valid set
            legal_valid_indices = []
            illegal_valid_indices = []
            for i, orig_idx in enumerate(valid_indices.cpu().numpy()):
                if orig_idx in legal_actions_set:
                    legal_valid_indices.append(i)
                else:
                    illegal_valid_indices.append(i)
            
            # illegal mass loss: sum of probabilities on all illegal actions
            if illegal_valid_indices:
                illegal_probs = probs[illegal_valid_indices]
                illegal_mass = illegal_probs.sum()
                illegal_mass_sum += illegal_mass.item()
                
                # Track Phase-1 illegal mass separately
                if is_phase1:
                    phase1_illegal_mass_sum += illegal_mass.item()
                    phase1_set_based_count += 1
                
                # Apply phase-specific multiplier for set-based losses (Phase-1 needs stronger signal)
                set_multiplier = phase1_set_based_multiplier if is_phase1 else 1.0
                
                # linear + squared penalty
                illegal_mass_loss = (illegal_mass_alpha * illegal_mass + 
                                    illegal_mass_beta * (illegal_mass ** 2)) * set_multiplier
                set_based_losses.append(illegal_mass_loss)
                
                # top-K illegal loss: penalize top-K illegal actions by probability
                # L_topk = δ · sum over top-K of −log(1 − p_illegal_k)
                if len(illegal_valid_indices) > 0:
                    topk_k = min(topk_illegal_k, len(illegal_valid_indices))
                    topk_illegal_probs, _ = torch.topk(illegal_probs, topk_k)
                    topk_illegal_sum += topk_illegal_probs.sum().item()
                    # compute −log(1 − p) for each top-K illegal action, then sum
                    epsilon = 1e-8
                    topk_illegal_probs_clamped = torch.clamp(topk_illegal_probs, min=epsilon, max=1.0 - epsilon)
                    topk_log_penalties = -torch.log1p(-topk_illegal_probs_clamped)  # -log(1 - p)
                    topk_loss = topk_illegal_delta * topk_log_penalties.sum() * set_multiplier
                    set_based_losses.append(topk_loss)
            
            # legal mass bonus: reward high probability on legal actions
            if legal_valid_indices:
                legal_probs = probs[legal_valid_indices]
                legal_mass = legal_probs.sum()
                legal_mass_sum += legal_mass.item()
                # bonus = -zeta * log(legal_mass + epsilon) to encourage high legal mass
                epsilon = 1e-8
                set_multiplier = phase1_set_based_multiplier if is_phase1 else 1.0
                legal_bonus = -legal_mass_bonus_zeta * torch.log(legal_mass + epsilon) * set_multiplier
                set_based_losses.append(legal_bonus)
            
            set_based_count += 1
        
        # compute standard loss (positive/negative example loss)
        # Apply phase-specific loss weights
        phase_weight = phase1_loss_weight if is_phase1 else phase0_loss_weight
        
        if is_neg:
            # for negative examples: penalize high probability on the illegal action
            log_probs = F.log_softmax(valid_logits, dim=0)
            illegal_log_prob = log_probs[action_compact]
            illegal_prob = torch.exp(illegal_log_prob)
            illegal_prob = torch.clamp(illegal_prob, min=1e-8, max=1.0 - 1e-8)
            log_penalty = -torch.log1p(-illegal_prob)
            squared_penalty = illegal_prob ** 2
            loss = (log_penalty + squared_penalty) * negative_loss_weight * phase_weight
        else:
            # for positive examples: standard cross-entropy
            base_loss = F.cross_entropy(valid_logits.unsqueeze(0), torch.tensor([action_compact], device=obs.device))
            
            # apply context-aware reward weighting if enabled
            if use_context_aware_reward_weighting and rewards is not None and b < len(rewards):
                reward = rewards[b].item() if isinstance(rewards, torch.Tensor) else rewards[b]
                grid_density = grid_densities[b].item() if grid_densities is not None and b < len(grid_densities) else None
                step_num = step_nums[b].item() if step_nums is not None and b < len(step_nums) else None
                
                # categorize game state: early-game (dense) vs late-game (sparse)
                # use both grid density and trajectory position for robustness
                is_early_game = True
                if grid_density is not None:
                    is_early_game = is_early_game and (grid_density > context_aware_early_threshold)
                if step_num is not None:
                    is_early_game = is_early_game and (step_num < context_aware_trajectory_threshold)
                
                # compute context-aware weight
                # for early-game: normalize by max reward in early-game category
                # for late-game: normalize by max reward in late-game category
                # we'll compute max rewards per category from the batch
                if grid_densities is not None and step_nums is not None:
                    # find max reward in the same category within this batch
                    category_rewards = []
                    for i in range(obs.size(0)):
                        if i < len(rewards) and i < len(grid_densities) and i < len(step_nums):
                            other_density = grid_densities[i].item() if isinstance(grid_densities, torch.Tensor) else grid_densities[i]
                            other_step = step_nums[i].item() if isinstance(step_nums, torch.Tensor) else step_nums[i]
                            other_reward = rewards[i].item() if isinstance(rewards, torch.Tensor) else rewards[i]
                            
                            other_is_early = (other_density > context_aware_early_threshold) and (other_step < context_aware_trajectory_threshold)
                            if other_is_early == is_early_game:
                                category_rewards.append(other_reward)
                    
                    if category_rewards:
                        max_reward_in_category = max(category_rewards)
                        # weight = reward / max_reward_in_category (normalized to [0, 1])
                        # add small epsilon to avoid division by zero
                        reward_weight = reward / (max_reward_in_category + 1e-8)
                        # clamp to reasonable range [0.1, 2.0] to avoid extreme weights
                        reward_weight = max(0.1, min(2.0, reward_weight))
                    else:
                        # fallback: use reward directly (normalized by max in batch)
                        max_reward_in_batch = max([r.item() if isinstance(r, torch.Tensor) else r for r in rewards[:obs.size(0)]])
                        reward_weight = reward / (max_reward_in_batch + 1e-8)
                        reward_weight = max(0.1, min(2.0, reward_weight))
                else:
                    # fallback: simple normalization by max reward in batch
                    if isinstance(rewards, torch.Tensor):
                        max_reward_in_batch = rewards[:obs.size(0)].max().item()
                    else:
                        max_reward_in_batch = max(rewards[:obs.size(0)])
                    reward_weight = reward / (max_reward_in_batch + 1e-8)
                    reward_weight = max(0.1, min(2.0, reward_weight))
                
                loss = base_loss * reward_weight * phase_weight
            else:
                loss = base_loss * phase_weight
            
        losses.append(loss)
        # Track phase-specific losses
        if is_phase1:
            phase1_losses.append(loss.item())
        else:
            phase0_losses.append(loss.item())
        
        # Get predicted action
        pred_action_compact = valid_logits.argmax().item()
        pred_action_original = valid_indices[pred_action_compact].item()
        
        # Track negative example accuracy (for negative examples, we want model to NOT pick the illegal action)
        if is_neg:
            negative_total += 1
            if pred_action_original != action:
                negative_correct += 1
        else:
            total += 1
            # For positive examples, we only track legal accuracy (not exact match)
            # because there are multiple valid moves per grid state
            
            # Phase-specific legal accuracy tracking (PRIMARY metric)
            if is_phase1:
                # Phase-1: extent selection
                phase1_total += 1
                # Legal accuracy: check if predicted extent is legal (sum=10)
                if legal_actions_set is not None:
                    if pred_action_original in legal_actions_set:
                        phase1_legal_correct += 1
                else:
                    # During curriculum, all exposed actions are legal
                    phase1_legal_correct += 1
            else:
                # Phase-0: anchor selection
                phase0_total += 1
                # Legal accuracy: check if predicted anchor has at least one legal extent
                if legal_actions_set is not None:
                    if pred_action_original in legal_actions_set:
                        phase0_legal_correct += 1
                else:
                    # During curriculum, all exposed actions are legal
                    phase0_legal_correct += 1

        # Track overall legal prediction count
        total_prediction_count += 1
        if legal_actions_set is not None:
            if pred_action_original in legal_actions_set:
                legal_prediction_count += 1
        else:
            # during curriculum phase we only expose legal actions, so treat as legal
            legal_prediction_count += 1
    
    # combine standard losses, set-based losses, and sum prediction loss
    if len(losses) == 0:
        loss = torch.tensor(0.0, device=obs.device, requires_grad=True)
    else:
        standard_loss = torch.stack(losses).mean()
        if set_based_losses:
            set_based_loss = torch.stack(set_based_losses).mean()
            loss = standard_loss + set_based_loss
        else:
            loss = standard_loss
        
        # Add sum prediction loss
        loss = loss + sum_prediction_loss_weight * sum_pred_loss
    
    # Note: We don't track exact match accuracy for positive examples
    # because there are multiple valid moves per grid state
    # Legal accuracy is the meaningful metric
    negative_accuracy = negative_correct / negative_total if negative_total > 0 else 0.0
    
    # compute average metrics
    avg_illegal_mass = illegal_mass_sum / set_based_count if set_based_count > 0 else 0.0
    avg_topk_illegal = topk_illegal_sum / set_based_count if set_based_count > 0 else 0.0
    avg_legal_mass = legal_mass_sum / set_based_count if set_based_count > 0 else 0.0
    
    # Phase-specific legal accuracies (PRIMARY metrics)
    phase0_legal_accuracy = phase0_legal_correct / phase0_total if phase0_total > 0 else 0.0
    phase1_legal_accuracy = phase1_legal_correct / phase1_total if phase1_total > 0 else 0.0
    
    # Phase-specific illegal mass (for Phase-1 debugging)
    phase1_illegal_mass = phase1_illegal_mass_sum / phase1_set_based_count if phase1_set_based_count > 0 else 0.0
    
    # Phase-specific average losses
    avg_phase0_loss = np.mean(phase0_losses) if phase0_losses else 0.0
    avg_phase1_loss = np.mean(phase1_losses) if phase1_losses else 0.0
    
    # Average entropy
    avg_entropy = np.mean(entropies) if entropies else 0.0
    
    info = {
        'loss': loss.item(),
        'negative_accuracy': negative_accuracy,  # For negative examples: did model avoid the illegal action?
        'positive_count': total,
        'negative_count': negative_total,
        'illegal_mass': avg_illegal_mass,
        'topk_illegal': avg_topk_illegal,
        'legal_mass': avg_legal_mass,
        'legal_predictions': legal_prediction_count,
        'total_predictions': total_prediction_count,
        'sum_prediction_loss': sum_pred_loss.item() if isinstance(sum_pred_loss, torch.Tensor) else sum_pred_loss,
        'sum_prediction_mae': mean_sum_error,
        # Phase-specific legal accuracies (PRIMARY metrics)
        'phase0_legal_accuracy': phase0_legal_accuracy,  # Does predicted anchor have valid extents?
        'phase0_count': phase0_total,
        'phase1_legal_accuracy': phase1_legal_accuracy,  # Does predicted extent sum to 10?
        'phase1_count': phase1_total,
        # Phase-specific losses and metrics
        'phase0_loss': avg_phase0_loss,
        'phase1_loss': avg_phase1_loss,
        'phase1_illegal_mass': phase1_illegal_mass,  # Probability mass on illegal extents in Phase-1
        # Entropy
        'entropy': avg_entropy,
    }
    
    return loss, info

