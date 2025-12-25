"""Logging utilities for SFT training."""
import torch
import torch.nn as nn
from typing import Dict, List

from rl.train.sft_utils import flat_idx_to_anchor, flat_idx_to_extent


def log_example_moves(
    policy: nn.Module,
    obs: torch.Tensor,
    actions: torch.Tensor,
    masks: torch.Tensor,
    batch_data: List[Dict],
    epoch: int,
    device: torch.device,
    num_examples: int = 5,
):
    """Log example moves predicted by the model"""
    policy.eval()
    with torch.no_grad():
        logits, _, _ = policy(obs, masks)  # ignore value and sum_predictions for logging
        
        examples_logged = 0
        for i in range(min(num_examples, len(batch_data))):
            if examples_logged >= num_examples:
                break
                
            data_item = batch_data[i]
            mask = masks[i]
            
            # handle sparse masks correctly (same as in compute_sft_loss)
            valid_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)  # [valid_count]
            valid_count = valid_indices.numel()
            
            if valid_count == 0:
                continue
            
            # extract valid logits (only at positions where mask is True)
            valid_logits = logits[i][valid_indices]  # [valid_action_count]
            
            # get predictions (compact index)
            pred_action_compact = valid_logits.argmax().item()
            pred_action_original = valid_indices[pred_action_compact].item()
            
            # get true action (original index)
            true_action = actions[i].item()
            
            # determine phase based on data structure
            is_phase0 = 'anchor' not in data_item
            
            if is_phase0:
                # phase-0: anchor selection
                pred_r1, pred_c1 = flat_idx_to_anchor(pred_action_original)
                true_r1, true_c1 = flat_idx_to_anchor(true_action)
                
                move_str = f"Phase-0: Predicted anchor=({pred_r1},{pred_c1}), True=({true_r1},{true_c1})"
            else:
                # phase-1: extent selection
                anchor_idx = data_item['anchor'].item()
                anchor_r1, anchor_c1 = flat_idx_to_anchor(anchor_idx)
                pred_r2, pred_c2 = flat_idx_to_extent(anchor_r1, anchor_c1, pred_action_original)
                true_r2, true_c2 = flat_idx_to_extent(anchor_r1, anchor_c1, true_action)
                
                move_str = f"Phase-1: Anchor=({anchor_r1},{anchor_c1}), Predicted extent=({pred_r2},{pred_c2}), True=({true_r2},{true_c2})"
            
            print(f"  Example {examples_logged + 1}: {move_str}")
            examples_logged += 1
        
        policy.train()

