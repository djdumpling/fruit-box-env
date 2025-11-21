"""SFT training script - main entry point.

Note: Sum prediction head pre-training should be done separately using:
    python rl/train/pretrain_sum_prediction.py
    
Then load the pre-trained checkpoint using --init_checkpoint when running this script.
"""
import sys
from pathlib import Path
# add project root to path for imports (go up 2 levels from rl/train/train_sft.py)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import os
import random
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from tqdm import tqdm
import wandb

from rl.models.policy import CNNPolicy
from rl.train.sft_config import Config
from rl.train.sft_utils import flat_idx_to_extent
from rl.train.sft_dataset import load_and_process_dataset
from rl.train.sft_negatives import generate_negatives_for_positive
from rl.train.sft_loss import compute_sft_loss
from rl.train.sft_logging import log_example_moves


def train(config: Config):
    """Main training loop"""
    # initialize wandb (always enabled)
    os.environ["WANDB_DIR"] = tempfile.gettempdir()
    wandb.init(
        project="fruit-box-sft",
        name=f"sft_seed{config.seed}",
        config={
            "dataset_name": config.dataset_name,
            "dataset_split": config.dataset_split,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
            "include_negative_examples": config.include_negative_examples,
            "negative_example_ratio": config.negative_example_ratio,
            "negative_loss_weight": config.negative_loss_weight,
            "illegal_mass_alpha": config.illegal_mass_alpha,
            "illegal_mass_beta": config.illegal_mass_beta,
            "topk_illegal_k": config.topk_illegal_k,
            "topk_illegal_delta": config.topk_illegal_delta,
            "legal_mass_bonus_zeta": config.legal_mass_bonus_zeta,
            "use_curriculum": config.use_curriculum,
            "curriculum_legal_only_epochs": config.curriculum_legal_only_epochs,
            "extent_curriculum_epochs": config.extent_curriculum_epochs,
            "min_extent_size": config.min_extent_size,
            "max_extent_size_early": config.max_extent_size_early,
            "use_reward_weighted_sampling": config.use_reward_weighted_sampling,
            "reward_sampling_alpha": config.reward_sampling_alpha,
            "use_context_aware_reward_weighting": config.use_context_aware_reward_weighting,
            "context_aware_early_threshold": config.context_aware_early_threshold,
            "context_aware_trajectory_threshold": config.context_aware_trajectory_threshold,
        },
        tags=["sft", "fruit-box", "supervised", "set-based-losses"],
    )
    print("Wandb initialized!")
    
    # set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # setup device (prefer CUDA, then CPU - skip MPS due to performance issues)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device} | Seed: {config.seed}")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # load and process dataset
    print("Loading and processing dataset...")
    phase0_data, phase1_data = load_and_process_dataset(
        config.dataset_name,
        config.dataset_split,
        seed=config.seed,
        include_negative_examples=config.include_negative_examples,
        negative_example_ratio=config.negative_example_ratio,
        extra_jsonl=config.extra_jsonl,
    )
    
    # create model with dropout for regularization
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170, dropout=config.dropout).to(device)
    if config.init_checkpoint:
        state_dict = torch.load(config.init_checkpoint, map_location=device)
        policy.load_state_dict(state_dict)
        print(f"Model initialized from checkpoint: {config.init_checkpoint}")
    else:
        print("Model created from scratch")
    
    # create optimizer with separate learning rates for Phase-0 and Phase-1
    # Phase-0 parameters: feature extractor, phase0_head, value_head, sum_prediction_head
    # Phase-1 parameters: phase1_head, anchor_embedding
    phase0_params = []
    phase1_params = []
    
    for name, param in policy.named_parameters():
        if 'phase1_head' in name or 'anchor_embedding' in name:
            phase1_params.append(param)
        else:
            phase0_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': phase0_params, 'lr': config.lr, 'weight_decay': config.weight_decay},
        {'params': phase1_params, 'lr': config.lr * config.phase1_lr_multiplier, 'weight_decay': config.weight_decay}
    ])
    print(f"Optimizer: Phase-0 LR={config.lr:.2e}, Phase-1 LR={config.lr * config.phase1_lr_multiplier:.2e} (multiplier={config.phase1_lr_multiplier})")
    
    # Note: Sum prediction pre-training should be done separately using pretrain_sum_prediction.py
    # If you want to use a pre-trained checkpoint, pass it via --init_checkpoint
    if config.init_checkpoint:
        print(f"Using pre-trained checkpoint: {config.init_checkpoint}")
        print("  (If this was from pretrain_sum_prediction.py, the sum prediction head is already trained)")
    
    # Main training loop
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        
        # curriculum learning: use legal-only masks for first N epochs
        # Phase-0 and Phase-1 have separate legal-only periods
        use_legal_only_masks_phase0 = (config.use_curriculum and 
                               epoch < config.curriculum_legal_only_epochs)
        use_legal_only_masks_phase1 = (config.use_curriculum and 
                                      epoch < config.curriculum_phase1_legal_only_epochs)
        
        if use_legal_only_masks_phase0 or use_legal_only_masks_phase1:
            phase0_status = f"legal-only (epoch {epoch + 1} < {config.curriculum_legal_only_epochs})" if use_legal_only_masks_phase0 else "all-geometric"
            phase1_status = f"legal-only (epoch {epoch + 1} < {config.curriculum_phase1_legal_only_epochs})" if use_legal_only_masks_phase1 else "all-geometric"
            print(f"  Curriculum: Phase-0={phase0_status}, Phase-1={phase1_status}")
        else:
            print(f"  Curriculum: Using all-geometric masks with set-based losses for both phases")
        
        # combine Phase-0 and Phase-1 positive examples only
        all_positive_data = phase0_data + phase1_data
        
        # apply turn-aware curriculum filtering
        if config.turn_based_curriculum:
            # Gradually include late-game examples (turn >= 25)
            if epoch < config.turn_curriculum_epochs:
                # Compute progress: 0.0 at epoch 0, 1.0 at turn_curriculum_epochs
                turn_curriculum_progress = min(1.0, epoch / max(config.turn_curriculum_epochs, 1))
                # Gradually include late-game examples
                filtered_data = []
                for example in all_positive_data:
                    step_num = example.get('step_num', 0)
                    if step_num < config.turn_threshold:
                        # Early-game examples: always include
                        filtered_data.append(example)
                    else:
                        # Late-game examples: include based on progress
                        if random.random() < turn_curriculum_progress:
                            filtered_data.append(example)
                all_positive_data = filtered_data
                if epoch == 0 or epoch % 5 == 0:
                    early_count = sum(1 for e in all_positive_data if e.get('step_num', 0) < config.turn_threshold)
                    late_count = len(all_positive_data) - early_count
                    print(f"  Turn-aware curriculum: {early_count} early-game (turn<{config.turn_threshold}), {late_count} late-game examples")
        
        # apply reward-weighted sampling if enabled
        if config.use_reward_weighted_sampling:
            # compute sampling weights: reward^alpha, but balance with trajectory position
            # to avoid over-sampling late-game moves
            weights = []
            for example in all_positive_data:
                reward = example.get('reward', 1)
                step_num = example.get('step_num', 0)
                
                # reward weight: higher reward = higher weight
                reward_weight = (reward + 1) ** config.reward_sampling_alpha  # +1 to avoid 0 weight
                
                # trajectory position weight: balance early/late game
                # early-game (step < threshold): weight = 1.0
                # late-game (step >= threshold): weight = 2.0 (oversample to compensate for rarity)
                if step_num < config.context_aware_trajectory_threshold:
                    position_weight = 1.0
                else:
                    position_weight = 2.0  # oversample late-game moves
                
                # combined weight: reward-weighted but balanced by position
                combined_weight = reward_weight * position_weight
                weights.append(combined_weight)
            
            # normalize weights to probabilities
            total_weight = sum(weights)
            if total_weight > 0:
                probabilities = [w / total_weight for w in weights]
                # sample with replacement using weights (for each epoch, we want to see high-reward examples more)
                # but we'll still iterate through all examples, just with weighted selection
                sampled_indices = np.random.choice(
                    len(all_positive_data),
                    size=len(all_positive_data),  # same size, but weighted
                    replace=True,
                    p=probabilities
                )
                all_positive_data = [all_positive_data[i] for i in sampled_indices]
                # shuffle after sampling so slices remain well-mixed and non-empty
                random.shuffle(all_positive_data)
                print(f"  Applied reward-weighted sampling (alpha={config.reward_sampling_alpha}) with trajectory balancing")
            else:
                random.shuffle(all_positive_data)
        else:
            random.shuffle(all_positive_data)
        
        # schedule legality and curriculum settings before epoch loop
        def interp(start: float, end: float, progress: float) -> float:
            return start + (end - start) * progress
        
        # Extent curriculum: delay expansion for first N epochs
        if config.extent_curriculum_epochs <= 0:
            extent_curriculum_progress = 1.0
        else:
            if epoch < config.extent_curriculum_delay_epochs:
                # Keep at early size for first N epochs
                extent_curriculum_progress = 0.0
            else:
                # Start expansion after delay
                effective_epoch = epoch - config.extent_curriculum_delay_epochs
                effective_curriculum_epochs = config.extent_curriculum_epochs - config.extent_curriculum_delay_epochs
                extent_curriculum_progress = min(
                    1.0,
                    effective_epoch / max(effective_curriculum_epochs, 1),
                )
        current_max_extent_size = int(round(interp(
            config.max_extent_size_early,
            config.extent_curriculum_final_size,
            extent_curriculum_progress,
        )))
        current_max_extent_size = max(current_max_extent_size, config.max_extent_size_early)
        
        # Gradual illegal exposure: start earlier to align with mask transition
        # Begin negative introduction when mask transition starts (epoch 15)
        transition_start = config.phase1_mask_transition_start_epoch
        if epoch < transition_start:
            # Before transition: no negative examples
            current_negative_ratio = 0.0
        else:
            # During and after transition: gradually increase negative ratio
            # Compute progress from transition start
            epochs_since_transition = max(0, epoch - transition_start + 1)
            # Use longer warmup period to align with mask transition
            total_warmup = config.phase1_mask_transition_epochs + config.negative_ratio_warmup_epochs
            if total_warmup <= 0:
                negative_ratio_progress = 1.0
            else:
                negative_ratio_progress = min(
                    1.0,
                    epochs_since_transition / max(total_warmup, 1),
                )
            current_negative_ratio = interp(
                config.negative_example_ratio_start,
                config.negative_example_ratio,
                negative_ratio_progress,
            )
        
        # Calculate gradual Phase-1 mask transition progress
        # For Phase-1: gradually transition from legal-only to all-geometric
        if epoch < config.phase1_mask_transition_start_epoch:
            phase1_mask_transition_progress = 0.0  # Fully legal-only
        elif epoch >= config.curriculum_phase1_legal_only_epochs:
            phase1_mask_transition_progress = 1.0  # Fully all-geometric
        else:
            # Gradual transition between start and end
            transition_epochs = config.curriculum_phase1_legal_only_epochs - config.phase1_mask_transition_start_epoch
            epochs_into_transition = epoch - config.phase1_mask_transition_start_epoch + 1
            phase1_mask_transition_progress = min(
                1.0,
                epochs_into_transition / max(transition_epochs, 1),
            )
        
        if config.loss_schedule_warmup_epochs <= 0:
            loss_schedule_progress = 1.0
        else:
            if epoch < config.loss_schedule_delay_epochs:
                loss_schedule_progress = 0.0
            else:
                warmed_up_epochs = epoch - config.loss_schedule_delay_epochs + 1
                loss_schedule_progress = min(
                    1.0,
                    warmed_up_epochs / max(config.loss_schedule_warmup_epochs, 1),
                )
        
        current_negative_loss_weight = interp(
            config.negative_loss_weight_start,
            config.negative_loss_weight,
            loss_schedule_progress,
        )
        current_illegal_mass_alpha = interp(
            config.illegal_mass_alpha_start,
            config.illegal_mass_alpha,
            loss_schedule_progress,
        )
        current_illegal_mass_beta = interp(
            config.illegal_mass_beta_start,
            config.illegal_mass_beta,
            loss_schedule_progress,
        )
        current_topk_illegal_delta = interp(
            config.topk_illegal_delta_start,
            config.topk_illegal_delta,
            loss_schedule_progress,
        )
        
        if config.sum_prediction_loss_warmup_epochs <= 0:
            sum_loss_progress = 1.0
        else:
            sum_loss_progress = min(
                1.0,
                (epoch + 1) / max(config.sum_prediction_loss_warmup_epochs, 1),
            )
        current_sum_pred_loss_weight = interp(
            config.sum_prediction_loss_start,
            config.sum_prediction_loss_weight,
            sum_loss_progress,
        )
        
        print(f"  Training on {len(all_positive_data)} positive examples ({len(phase0_data)} Phase-0 + {len(phase1_data)} Phase-1)")
        if config.include_negative_examples:
            print(f"  Generating negatives on-the-fly with ratio {current_negative_ratio:.2f}:1 "
                  f"(weight={current_negative_loss_weight:.2f}, schedule progress={loss_schedule_progress:.2f})")
        if config.extent_curriculum_epochs > 0:
            print(f"  Extent curriculum progress={extent_curriculum_progress:.2f} "
                  f"(max extent size={current_max_extent_size})")
        if not (use_legal_only_masks_phase0 and use_legal_only_masks_phase1):
            print(f"  Set-based weights this epoch → alpha={current_illegal_mass_alpha:.2f}, "
                  f"beta={current_illegal_mass_beta:.2f}, topk_delta={current_topk_illegal_delta:.2f}")
        print(f"  Sum-head loss weight={current_sum_pred_loss_weight:.3f} (progress={sum_loss_progress:.2f})")
        if phase1_mask_transition_progress > 0.0 and phase1_mask_transition_progress < 1.0:
            print(f"  Phase-1 mask transition: {phase1_mask_transition_progress:.2%} (gradual transition active)")
        
        policy.train()
        epoch_losses = []
        instrument_epoch = (
            config.instrument_batches and 
            epoch < config.instrument_batches_epochs
        )
        instrumentation_samples = []
        batch_data_for_logging = None
        batch_obs_for_logging = None
        batch_actions_for_logging = None
        batch_masks_for_logging = None
        
        # calculate batch composition: if ratio=1:1, batch_size=128, then ~64 positives, ~64 negatives
        if config.include_negative_examples:
            ratio = max(current_negative_ratio, 0.0)
            positive_per_batch = max(1, int(config.batch_size / (ratio + 1)))
            negative_per_batch = config.batch_size - positive_per_batch
        else:
            positive_per_batch = config.batch_size
            negative_per_batch = 0
        
        # statistics tracking
        hard_negative_count = 0
        total_negative_count = 0
        extent_sizes = []  # track max(dr, dc) for Phase-1 examples
        
        for batch_idx, start in enumerate(tqdm(range(0, len(all_positive_data), positive_per_batch), desc="Training")):
            # sample positive examples for this batch
            candidate_positives = all_positive_data[start:start + positive_per_batch]
            if not candidate_positives:
                continue  # skip empty slices (can occur due to sampling)
            
            # apply extent-size curriculum filtering (gradual expansion with turn-aware limits)
            batch_positives = []
            curriculum_active = (
                config.extent_curriculum_epochs > 0 and
                current_max_extent_size < config.extent_curriculum_final_size
            )
            if curriculum_active:
                for pos_example in candidate_positives:
                    if pos_example.get('phase') == 1:
                        r1 = pos_example['r1']
                        c1 = pos_example['c1']
                        action_idx = pos_example['action'].item()
                        r2, c2 = flat_idx_to_extent(r1, c1, action_idx)
                        dr = r2 - r1
                        dc = c2 - c1
                        max_size = max(dr, dc)
                        
                        # Turn-aware extent limits: early-game (turn < 25) uses stricter limits
                        step_num = pos_example.get('step_num', 0)
                        if config.turn_based_curriculum and step_num < config.turn_threshold:
                            # Early-game: use stricter limit
                            effective_max_size = min(current_max_extent_size, config.turn_early_max_extent_size)
                        elif config.turn_based_curriculum:
                            # Late-game: use more lenient limit
                            effective_max_size = min(current_max_extent_size, config.turn_late_max_extent_size)
                        else:
                            # No turn-aware curriculum: use standard limit
                            effective_max_size = current_max_extent_size
                        
                        if config.min_extent_size <= max_size <= effective_max_size:
                            batch_positives.append(pos_example)
                            extent_sizes.append(max_size)
                    else:
                        batch_positives.append(pos_example)
                
                next_idx = start + len(batch_positives)
                max_search = min(len(all_positive_data), start + positive_per_batch * 3)
                while len(batch_positives) < positive_per_batch and next_idx < max_search:
                    if next_idx < len(all_positive_data):
                        candidate = all_positive_data[next_idx]
                        if candidate.get('phase') == 1:
                            r1 = candidate['r1']
                            c1 = candidate['c1']
                            action_idx = candidate['action'].item()
                            r2, c2 = flat_idx_to_extent(r1, c1, action_idx)
                            dr = r2 - r1
                            dc = c2 - c1
                            max_size = max(dr, dc)
                            
                            # Turn-aware extent limits (same logic as above)
                            step_num = candidate.get('step_num', 0)
                            if config.turn_based_curriculum and step_num < config.turn_threshold:
                                effective_max_size = min(current_max_extent_size, config.turn_early_max_extent_size)
                            elif config.turn_based_curriculum:
                                effective_max_size = min(current_max_extent_size, config.turn_late_max_extent_size)
                            else:
                                effective_max_size = current_max_extent_size
                            
                            if config.min_extent_size <= max_size <= effective_max_size:
                                batch_positives.append(candidate)
                                extent_sizes.append(max_size)
                        else:
                            batch_positives.append(candidate)
                    next_idx += 1
                    if len(batch_positives) >= positive_per_batch:
                        break
            else:
                batch_positives = candidate_positives
                for pos_example in batch_positives:
                    if pos_example.get('phase') == 1:
                        r1 = pos_example['r1']
                        c1 = pos_example['c1']
                        action_idx = pos_example['action'].item()
                        r2, c2 = flat_idx_to_extent(r1, c1, action_idx)
                        dr = r2 - r1
                        dc = c2 - c1
                        max_size = max(dr, dc)
                        extent_sizes.append(max_size)
            
            # generate negative examples on-the-fly for each positive
            batch_negatives = []
            if config.include_negative_examples:
                for pos_example in batch_positives:
                    negs, neg_stats = generate_negatives_for_positive(pos_example, current_negative_ratio)
                    # track hard negative statistics
                    if neg_stats['used_hard_negatives']:
                        hard_negative_count += 1
                    total_negative_count += 1
                    # limit negatives per positive to maintain batch size
                    if len(batch_negatives) + len(negs) <= negative_per_batch:
                        batch_negatives.extend(negs)
                    else:
                        # take only what we need
                        remaining = negative_per_batch - len(batch_negatives)
                        batch_negatives.extend(negs[:remaining])
                        break
            
            if not batch_positives:
                continue  # nothing to train on this iteration

            # combine positives and negatives into batch
            batch_data = batch_positives + batch_negatives
            # shuffle to mix positives and negatives
            random.shuffle(batch_data)
            
            # extract legal actions sets for set-based losses
            legal_actions_sets = []
            for d in batch_data:
                if d.get('phase') == 0:
                    # Phase-0: legal anchors
                    legal_actions_sets.append(d.get('legal_anchors_set', set()))
                else:
                    # Phase-1: legal extents
                    legal_actions_sets.append(d.get('legal_extents_set', set()))
            
            # update masks based on curriculum learning (phase-specific)
            # Check phase for each example to apply phase-specific legal-only masks
            updated_masks = []
            for i, d in enumerate(batch_data):
                phase = d.get('phase', 0)
                
                if phase == 0:
                    # Phase-0: use standard logic
                    use_legal_only = use_legal_only_masks_phase0
                    if use_legal_only:
                        # curriculum phase: use legal-only masks
                        mask = torch.zeros(170, dtype=torch.bool)
                        legal_set = legal_actions_sets[i]
                        for legal_idx in legal_set:
                            if legal_idx < 170 and legal_idx > 0:
                                mask[legal_idx] = True
                        updated_masks.append(mask)
                    else:
                        # full phase: use all-geometric masks
                        mask = d["mask"] if d["mask"].shape[0] == 170 else torch.cat([d["mask"], torch.zeros(170 - d["mask"].shape[0], dtype=torch.bool)])
                        updated_masks.append(mask)
                else:
                    # Phase-1: use gradual transition
                    if phase1_mask_transition_progress < 1.0:
                        # Gradual transition: mix legal and illegal actions
                        mask = torch.zeros(170, dtype=torch.bool)
                        legal_set = legal_actions_sets[i]
                        
                        # Always include all legal actions
                        for legal_idx in legal_set:
                            if legal_idx < 170 and legal_idx > 0:
                                mask[legal_idx] = True
                        
                        # Gradually include illegal actions based on transition progress
                        if phase1_mask_transition_progress > 0.0:
                            # Get all geometrically valid extents (from original mask)
                            original_mask = d["mask"] if d["mask"].shape[0] == 170 else torch.cat([d["mask"], torch.zeros(170 - d["mask"].shape[0], dtype=torch.bool)])
                            illegal_indices = []
                            for idx in range(170):
                                if original_mask[idx] and idx not in legal_set and idx > 0:
                                    illegal_indices.append(idx)
                            
                            # Sample illegal actions to include based on transition progress
                            num_illegal_to_include = int(len(illegal_indices) * phase1_mask_transition_progress)
                            if num_illegal_to_include > 0 and illegal_indices:
                                sampled_illegal = random.sample(
                                    illegal_indices,
                                    min(num_illegal_to_include, len(illegal_indices))
                                )
                                for illegal_idx in sampled_illegal:
                                    mask[illegal_idx] = True
                        
                        updated_masks.append(mask)
                    else:
                        # Fully transitioned: use all-geometric masks
                        mask = d["mask"] if d["mask"].shape[0] == 170 else torch.cat([d["mask"], torch.zeros(170 - d["mask"].shape[0], dtype=torch.bool)])
                        updated_masks.append(mask)
            
            batch_masks = torch.stack(updated_masks).to(device)
            
            # stack batches
            batch_obs = torch.stack([d['obs'] for d in batch_data]).to(device)
            batch_actions = torch.stack([d['action'] for d in batch_data]).to(device)
            batch_is_positive = torch.tensor([d.get('is_positive', True) for d in batch_data], dtype=torch.bool).to(device)
            
            # extract reward and context information for context-aware weighting
            # negatives don't have reward/context (they're synthetic), so use defaults
            batch_rewards = None
            batch_grid_densities = None
            batch_step_nums = None
            if config.use_context_aware_reward_weighting:
                batch_rewards = torch.tensor([
                    d.get('reward', 0) if d.get('is_positive', True) else 0 
                    for d in batch_data
                ], dtype=torch.float32).to(device)
                batch_grid_densities = torch.tensor([
                    d.get('grid_density', 0.5) if d.get('is_positive', True) else 0.5 
                    for d in batch_data
                ], dtype=torch.float32).to(device)
                batch_step_nums = torch.tensor([
                    d.get('step_num', 0) if d.get('is_positive', True) else 0 
                    for d in batch_data
                ], dtype=torch.long).to(device)
            
            # forward pass with set-based losses (only when not in curriculum phase)
            # Use set-based losses when at least one phase is out of legal-only period
            # This allows Phase-0 to use set-based losses while Phase-1 is still in legal-only
            use_set_based = not (use_legal_only_masks_phase0 and use_legal_only_masks_phase1)
            loss, info = compute_sft_loss(
                policy, batch_obs, batch_actions, batch_masks, batch_is_positive,
                negative_loss_weight=current_negative_loss_weight,
                legal_actions_sets=legal_actions_sets,
                illegal_mass_alpha=current_illegal_mass_alpha,
                illegal_mass_beta=current_illegal_mass_beta,
                topk_illegal_k=config.topk_illegal_k,
                topk_illegal_delta=current_topk_illegal_delta,
                legal_mass_bonus_zeta=config.legal_mass_bonus_zeta,
                use_set_based_losses=use_set_based,
                rewards=batch_rewards,
                grid_densities=batch_grid_densities,
                step_nums=batch_step_nums,
                use_context_aware_reward_weighting=config.use_context_aware_reward_weighting,
                context_aware_early_threshold=config.context_aware_early_threshold,
                context_aware_trajectory_threshold=config.context_aware_trajectory_threshold,
                sum_prediction_loss_weight=current_sum_pred_loss_weight,
                phase0_loss_weight=config.phase0_loss_weight,
                phase1_loss_weight=config.phase1_loss_weight,
                phase1_set_based_multiplier=config.phase1_set_based_multiplier,
            )
            batch_loss_value = float(loss.detach().item())
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # clip_grad_norm_ returns the pre-clipped norm, but we want to log the post-clipped norm
            pre_clipped_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip_norm)
            # Compute actual post-clipped norm to verify clipping worked
            post_clipped_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float('inf'))
            optimizer.step()
            
            # Add gradient norm to info (post-clipped norm, should be <= grad_clip_norm)
            info['grad_norm'] = post_clipped_norm.item()
            
            epoch_losses.append(info)
            
            # Batch-level instrumentation for early epochs to diagnose exploding loss/gradients
            should_instrument = (
                instrument_epoch and 
                (batch_idx % max(config.instrument_batches_every, 1) == 0)
            )
            if should_instrument:
                positives_in_batch = int(batch_is_positive.sum().item()) if batch_is_positive is not None else len(batch_data)
                negatives_in_batch = len(batch_data) - positives_in_batch
                instrumentation_samples.append({
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                    "loss": batch_loss_value,
                    "grad_norm": post_clipped_norm.item(),
                    "phase0_legal": info.get('phase0_legal_accuracy', 0.0),
                    "phase1_legal": info.get('phase1_legal_accuracy', 0.0),
                    "negatives": negatives_in_batch,
                    "positives": positives_in_batch,
                    "using_set_losses": use_set_based,
                    "neg_ratio": current_negative_ratio,
                })
            
            # save first batch for logging example moves
            if batch_idx == 0:
                batch_data_for_logging = batch_data
                batch_obs_for_logging = batch_obs
                batch_actions_for_logging = batch_actions
                batch_masks_for_logging = batch_masks
        
        # logging
        avg_loss = np.mean([d['loss'] for d in epoch_losses])
        avg_negative_accuracy = np.mean([d.get('negative_accuracy', 0.0) for d in epoch_losses])
        total_legal_predictions = sum(d.get('legal_predictions', 0) for d in epoch_losses)
        total_predictions = sum(d.get('total_predictions', 0) for d in epoch_losses)
        avg_legality_rate = (total_legal_predictions / total_predictions) if total_predictions > 0 else 0.0
        total_positive = sum(d.get('positive_count', 0) for d in epoch_losses)
        total_negative = sum(d.get('negative_count', 0) for d in epoch_losses)
        
        # set-based loss metrics
        avg_illegal_mass = np.mean([d.get('illegal_mass', 0.0) for d in epoch_losses])
        avg_topk_illegal = np.mean([d.get('topk_illegal', 0.0) for d in epoch_losses])
        avg_legal_mass = np.mean([d.get('legal_mass', 0.0) for d in epoch_losses])
        
        # hard negative mining statistics
        hard_negative_ratio = hard_negative_count / max(total_negative_count, 1)
        
        # extent size distribution
        # Note: avg_extent_size is max(dr, dc), not the number of non-zero cells within the extent
        avg_extent_size = np.mean(extent_sizes) if extent_sizes else 0.0
        max_extent_size = max(extent_sizes) if extent_sizes else 0
        
        # Phase-specific legal accuracy metrics (PRIMARY)
        total_phase0 = sum([info.get('phase0_count', 0) for info in epoch_losses])
        total_phase1 = sum([info.get('phase1_count', 0) for info in epoch_losses])
        avg_phase0_legal_accuracy = np.mean([info.get('phase0_legal_accuracy', 0.0) for info in epoch_losses if info.get('phase0_count', 0) > 0] or [0.0])
        avg_phase1_legal_accuracy = np.mean([info.get('phase1_legal_accuracy', 0.0) for info in epoch_losses if info.get('phase1_count', 0) > 0] or [0.0])
        
        # Phase-specific losses and metrics
        avg_phase0_loss = np.mean([info.get('phase0_loss', 0.0) for info in epoch_losses if info.get('phase0_loss', 0.0) > 0] or [0.0])
        avg_phase1_loss = np.mean([info.get('phase1_loss', 0.0) for info in epoch_losses if info.get('phase1_loss', 0.0) > 0] or [0.0])
        avg_phase1_illegal_mass = np.mean([info.get('phase1_illegal_mass', 0.0) for info in epoch_losses if info.get('phase1_illegal_mass', 0.0) > 0] or [0.0])
        
        # Entropy and gradient norm (guard against NaN/Inf so Wandb keeps logging)
        avg_entropy = float(np.nan_to_num(
            np.mean([info.get('entropy', 0.0) for info in epoch_losses]),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ))
        avg_grad_norm = float(np.nan_to_num(
            np.mean([info.get('grad_norm', 0.0) for info in epoch_losses]),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ))
        
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Legality rate={avg_legality_rate:.4f}")
        print(f"  Phase-0: Legal accuracy={avg_phase0_legal_accuracy:.4f}, Loss={avg_phase0_loss:.4f} ({total_phase0} examples)")
        print(f"  Phase-1: Legal accuracy={avg_phase1_legal_accuracy:.4f}, Loss={avg_phase1_loss:.4f}, Illegal mass={avg_phase1_illegal_mass:.4f} ({total_phase1} examples)")
        print(f"  Entropy: {avg_entropy:.4f}, Grad norm: {avg_grad_norm:.4f}")
        if total_negative > 0:
            print(f"  Positive examples: {total_positive}, Negative examples: {total_negative}")
            print(f"  Negative accuracy (avoiding illegal actions): {avg_negative_accuracy:.4f}")
            print(f"  Hard negative mining: {hard_negative_count}/{total_negative_count} ({hard_negative_ratio:.2%})")
        if extent_sizes:
            print(f"  Extent size: avg={avg_extent_size:.2f}, max={max_extent_size}")
        if not (use_legal_only_masks_phase0 and use_legal_only_masks_phase1):
            print(f"  Set-based losses: Illegal mass={avg_illegal_mass:.4f}, Top-K illegal={avg_topk_illegal:.4f}, Legal mass={avg_legal_mass:.4f}")
        
        # log example moves
        if batch_data_for_logging is not None:
            print("  Example moves:")
            log_example_moves(
                policy,
                batch_obs_for_logging,
                batch_actions_for_logging,
                batch_masks_for_logging,
                batch_data_for_logging,
                epoch + 1,
                device,
                num_examples=5,
            )
        
        # Print instrumentation summary to help pinpoint instability onset
        if instrumentation_samples:
            print("  Instrumentation samples (early epochs):")
            for sample in instrumentation_samples[:10]:
                print(
                    f"    Batch {sample['batch']:03d} | "
                    f"loss={sample['loss']:.2f} | grad_norm={sample['grad_norm']:.2f} | "
                    f"phase0_legal={sample['phase0_legal']:.2f} | phase1_legal={sample['phase1_legal']:.2f} | "
                    f"positives={sample['positives']} neg={sample['negatives']} | "
                    f"neg_ratio={sample['neg_ratio']:.2f} | "
                    f"{'set-loss' if sample['using_set_losses'] else 'curriculum-only'}"
            )
        
        log_dict = {
            "epoch": epoch + 1,
            "train/loss": avg_loss,
            "train/legality_rate": avg_legality_rate,  # Overall legal action rate
            # Phase-specific legal accuracies (PRIMARY metrics)
            "train/phase0_legal_accuracy": avg_phase0_legal_accuracy,  # Valid anchor selection
            "train/phase1_legal_accuracy": avg_phase1_legal_accuracy,  # Valid extent selection
            "train/phase0_count": total_phase0,
            "train/phase1_count": total_phase1,
            # Phase-specific losses and metrics
            "train/phase0_loss": avg_phase0_loss,
            "train/phase1_loss": avg_phase1_loss,
            "train/phase1_illegal_mass": avg_phase1_illegal_mass,  # Probability mass on illegal extents in Phase-1
            # Training dynamics
            "train/entropy": avg_entropy,
            "train/grad_norm": avg_grad_norm,
            "train/loss_schedule_progress": loss_schedule_progress,
            "train/negative_loss_weight_active": current_negative_loss_weight,
            "train/illegal_mass_alpha_active": current_illegal_mass_alpha,
            "train/illegal_mass_beta_active": current_illegal_mass_beta,
            "train/topk_illegal_delta_active": current_topk_illegal_delta,
            "train/negative_ratio_active": current_negative_ratio,
            "train/sum_loss_weight_active": current_sum_pred_loss_weight,
            "train/sum_loss_progress": sum_loss_progress,
        }
        if total_negative > 0:
            log_dict["train/negative_accuracy"] = avg_negative_accuracy
            log_dict["train/positive_count"] = total_positive
            log_dict["train/negative_count"] = total_negative
            log_dict["train/hard_negative_ratio"] = hard_negative_ratio
        if extent_sizes:
            log_dict["train/avg_extent_size"] = avg_extent_size
            log_dict["train/max_extent_size"] = max_extent_size
            # log histogram of extent sizes
            if len(extent_sizes) > 0:
                log_dict["train/extent_size_hist"] = wandb.Histogram(extent_sizes)
        if not (use_legal_only_masks_phase0 and use_legal_only_masks_phase1):
            log_dict["train/illegal_mass"] = avg_illegal_mass
            log_dict["train/topk_illegal"] = avg_topk_illegal
            log_dict["train/legal_mass"] = avg_legal_mass
        wandb.log(log_dict)
        
        # checkpoint
        if (epoch + 1) % config.checkpoint_interval == 0:
            checkpoint_path = f"{config.checkpoint_dir}/policy_sft_epoch{epoch+1}.pt"
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            artifact = wandb.Artifact(
                name=f"sft-checkpoint-epoch-{epoch+1}",
                type="model",
                description=f"SFT checkpoint at epoch {epoch+1}",
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
    
    # save final checkpoint and weights
    final_checkpoint_path = f"{config.checkpoint_dir}/policy_sft_final.pt"
    final_weights_path = f"{config.checkpoint_dir}/policy_sft_final_weights.pt"
    
    torch.save(policy.state_dict(), final_checkpoint_path)
    torch.save(policy.state_dict(), final_weights_path)  # explicit weights file
    print(f"\nTraining complete!")
    print(f"  Final checkpoint: {final_checkpoint_path}")
    print(f"  Final weights: {final_weights_path}")
    
    artifact = wandb.Artifact(
        name="sft-checkpoint-final",
        type="model",
        description=f"Final SFT checkpoint after {config.epochs} epochs",
    )
    artifact.add_file(final_checkpoint_path)
    artifact.add_file(final_weights_path)
    wandb.log_artifact(artifact)
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dataset_name", type=str, default="djdumpling/fruit-box-minimal-area")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--extra_jsonl", type=str, default=None, help="Optional local JSONL with corrective data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=20)
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Optional initial checkpoint to warm start")
    parser.add_argument("--negative_example_ratio", type=float, default=2.0, help="Ratio of negative examples per positive")
    args = parser.parse_args()
    
    config = Config(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        extra_jsonl=args.extra_jsonl,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        init_checkpoint=args.init_checkpoint,
        negative_example_ratio=args.negative_example_ratio,
    )
    
    train(config)


if __name__ == "__main__":
    main()
