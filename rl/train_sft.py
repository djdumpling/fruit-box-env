"""
Supervised Fine-Tuning (SFT) for Fruit Box environment.

This script loads trajectories from the Hugging Face dataset and trains a CNNPolicy
model using supervised learning. The trained model can then be used as a starting
point for GRPO training in train_grpo.py.

The model architecture matches train_grpo.py exactly, so checkpoints are compatible.
To load an SFT checkpoint in train_grpo.py, modify the code to load the checkpoint
before training starts:
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    policy.load_state_dict(torch.load("checkpoints/policy_sft_final.pt", map_location=device))

Usage:
    python rl/train_sft.py --seed 42 --epochs 200 --batch_size 64 --lr 1e-4
"""

import sys
from pathlib import Path
# add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
import random
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from datasets import load_dataset
import wandb

from rl.models.policy import CNNPolicy
from fruit_box import Sum10Env


@dataclass
class Config:
    """SFT training configuration"""
    # data
    dataset_name: str = "djdumpling/fruit-box-minimal-area"
    dataset_split: str = "train"
    
    # training
    epochs: int = 200
    batch_size: int = 128  # increased for more stable gradients
    lr: float = 2e-4  # increased learning rate for faster convergence
    weight_decay: float = 1e-5
    
    # negative examples (for learning legality)
    include_negative_examples: bool = True
    negative_example_ratio: float = 1.0  # ratio of negative to positive examples (1.0 = 1 negative per positive on average)
    # note: real-world ratio is ~41:1 illegal to legal, but we use 1.0 because:
    # - positive examples are more informative (show correct actions)
    # - negative_loss_weight (2.0) already emphasizes negative examples
    # - too many negatives can make model overly conservative
    negative_loss_weight: float = 2.0  # weight for negative example loss (higher = more emphasis on avoiding illegal actions)
    
    # other
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 2


def anchor_to_flat_idx(r1: int, c1: int) -> int:
    """Convert anchor (r1, c1) to flat index [0, 169]"""
    return r1 * 17 + c1


def flat_idx_to_anchor(idx: int) -> Tuple[int, int]:
    """Convert flat index [0, 169] to anchor (r1, c1)"""
    r1 = idx // 17
    c1 = idx % 17
    return (r1, c1)


def extent_to_flat_idx(r1: int, c1: int, r2: int, c2: int) -> int:
    """Convert extent (r2, c2) to flat index given anchor (r1, c1)
    
    Valid extents: r2 in [r1, 9], c2 in [c1, 16]
    Flat index: (r2 - r1) * (17 - c1) + (c2 - c1)
    """
    if not (r1 <= r2 < 10 and c1 <= c2 < 17):
        raise ValueError(f"Invalid extent: anchor=({r1},{c1}), extent=({r2},{c2})")
    dr = r2 - r1
    dc = c2 - c1
    width = 17 - c1
    return dr * width + dc


def flat_idx_to_extent(r1: int, c1: int, idx: int) -> Tuple[int, int]:
    """Convert flat index to extent (r2, c2) given anchor (r1, c1)"""
    width = 17 - c1
    dr = idx // width
    dc = idx % width
    r2 = r1 + dr
    c2 = c1 + dc
    return (r2, c2)


def build_observation(grid: np.ndarray, phase: int, selected_anchor: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Build 4-channel observation from grid
    
    phase=0 for anchor selection, phase=1 for extent selection.
    selected_anchor only used in phase 1.
    """
    grid = grid.astype(np.float32)
    
    # channel 0: normalized values
    value_norm = grid / 9.0
    
    # channel 1: nonzero mask
    nonzero_mask = (grid > 0).astype(np.float32)
    
    # channel 2: anchor mask (zeros in Phase-0, selected anchor=1 in Phase-1)
    anchor_mask = np.zeros((10, 17), dtype=np.float32)
    if phase == 1 and selected_anchor is not None:
        r1, c1 = selected_anchor
        anchor_mask[r1, c1] = 1.0
    
    # channel 3: phase mask (all zeros in Phase-0, all ones in Phase-1)
    phase_mask = np.full((10, 17), float(phase), dtype=np.float32)
    
    obs = np.stack([value_norm, nonzero_mask, anchor_mask, phase_mask], axis=0)
    return obs


def get_grid_hash(grid: np.ndarray) -> bytes:
    """Get hashable representation of grid for caching"""
    return grid.tobytes()


def compute_legal_anchors(grid: np.ndarray) -> set:
    """Find all anchors that have at least one legal extent"""
    temp_env = Sum10Env()
    temp_env.reset(grid=grid.copy())
    
    legal_anchors_set = set()
    for anchor_r1 in range(10):
        for anchor_c1 in range(17):
            anchor_idx = anchor_to_flat_idx(anchor_r1, anchor_c1)
            # check if this anchor has any legal extents
            max_valid_count = (10 - anchor_r1) * (17 - anchor_c1)
            has_legal = False
            for extent_idx in range(max_valid_count):
                r2_test, c2_test = flat_idx_to_extent(anchor_r1, anchor_c1, extent_idx)
                if temp_env.box_sum(anchor_r1, anchor_c1, r2_test, c2_test) == 10:
                    reward_test = temp_env.box_nonzero_count(anchor_r1, anchor_c1, r2_test, c2_test)
                    if reward_test > 0:
                        has_legal = True
                        break
            if has_legal:
                legal_anchors_set.add(anchor_idx)
    
    return legal_anchors_set


def compute_legal_extents(grid: np.ndarray, r1: int, c1: int) -> set:
    """Find all legal extents for a given anchor"""
    temp_env = Sum10Env()
    temp_env.reset(grid=grid.copy())
    
    legal_extents_set = set()
    max_valid_count = (10 - r1) * (17 - c1)
    for extent_idx in range(max_valid_count):
        r2_test, c2_test = flat_idx_to_extent(r1, c1, extent_idx)
        # check if this extent sums to 10
        if temp_env.box_sum(r1, c1, r2_test, c2_test) == 10:
            reward_test = temp_env.box_nonzero_count(r1, c1, r2_test, c2_test)
            if reward_test > 0:  # Must clear at least one cell
                legal_extents_set.add(extent_idx)
    
    return legal_extents_set


def compute_illegal_anchors(grid: np.ndarray, legal_anchors_set: set) -> set:
    """Find all anchors that DON'T have any legal extents"""
    all_anchors = set(range(170))
    illegal_anchors_set = all_anchors - legal_anchors_set
    return illegal_anchors_set


def compute_illegal_extents(grid: np.ndarray, r1: int, c1: int, legal_extents_set: set) -> set:
    """Find all geometrically valid extents that DON'T sum to 10"""
    max_valid_count = (10 - r1) * (17 - c1)
    all_extents = set(range(max_valid_count))
    illegal_extents_set = all_extents - legal_extents_set
    return illegal_extents_set


def load_and_process_dataset(
    dataset_name: str,
    dataset_split: str,
    seed: Optional[int] = None,
    include_negative_examples: bool = True,
    negative_example_ratio: float = 0.5,
) -> Tuple[List[Dict], List[Dict]]:
    """Load dataset and convert to Phase-0/Phase-1 examples
    
    If include_negative_examples=True, generates negative examples (illegal anchors/extents)
    to teach the policy which actions are invalid.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    hf_dataset = load_dataset(dataset_name, split=dataset_split)
    print(f"Loaded dataset {dataset_name} (split: {dataset_split})...")
    
    # group trajectories by episode_id and agent_tag
    episodes = {}
    for row in hf_dataset:
        ep_id = row["episode_id"]
        agent_tag = row.get("agent_tag", "unknown")
        key = f"{ep_id}_{agent_tag}"
        if key not in episodes:
            episodes[key] = []
        episodes[key].append(row)
    
    for key in episodes:
        episodes[key].sort(key=lambda x: x["step"])
    
    phase0_data = []
    phase1_data = []
    
    # performance optimization: cache legal anchors/extents per grid state
    legal_anchors_cache = {}  # grid_hash -> set of legal anchor indices
    legal_extents_cache = {}  # (grid_hash, anchor_idx) -> set of legal extent indices
    
    # debug: track first few examples
    debug_count = 0
    max_debug_examples = 3
    
    # count total steps for progress tracking
    total_steps = sum(len(trajectory) for trajectory in episodes.values())
    print(f"Processing {total_steps} trajectory steps...")
    
    # process each trajectory
    processed_steps = 0
    for key, trajectory in tqdm(episodes.items(), desc="Processing trajectories", unit="traj"):
        if not trajectory:
            continue
        
        # process each step in the trajectory
        for step in trajectory:
            processed_steps += 1
            # extract grid directly from dataset
            grid = np.array(step["grid"], dtype=np.uint8)
            
            # extract action coordinates
            action = step.get("action", {})
            r1 = action.get("r1", -1)
            c1 = action.get("c1", -1)
            r2 = action.get("r2", -1)
            c2 = action.get("c2", -1)
            
            # skip invalid actions
            if r1 == -1 or c1 == -1 or r2 == -1 or c2 == -1:
                continue
            
            # validate coordinates
            if not (0 <= r1 < 10 and 0 <= c1 < 17 and 0 <= r2 < 10 and 0 <= c2 < 17):
                print(f"Warning: Invalid coordinates - r1={r1}, c1={c1}, r2={r2}, c2={c2}")
                continue
            
            # validate extent is valid (r2 >= r1, c2 >= c1)
            if not (r1 <= r2 and c1 <= c2):
                print(f"Warning: Invalid extent - anchor=({r1},{c1}), extent=({r2},{c2})")
                continue
            
            # debug output for first few examples
            if debug_count < max_debug_examples:
                print(f"\n[DEBUG] Example {debug_count + 1}:")
                print(f"  Grid shape: {grid.shape}")
                print(f"  Grid sample (first row): {grid[0, :5].tolist()}...")
                print(f"  Action: r1={r1}, c1={c1}, r2={r2}, c2={c2}")
                print(f"  Anchor flat idx: {anchor_to_flat_idx(r1, c1)}")
                print(f"  Extent flat idx: {extent_to_flat_idx(r1, c1, r2, c2)}")
                # verify round-trip conversion
                recovered_r2, recovered_c2 = flat_idx_to_extent(r1, c1, extent_to_flat_idx(r1, c1, r2, c2))
                print(f"  Round-trip check: ({r2},{c2}) -> {extent_to_flat_idx(r1, c1, r2, c2)} -> ({recovered_r2},{recovered_c2})")
                debug_count += 1
            
            # phase-0: select anchor (r1, c1)
            # only include anchors that have at least one legal extent
            # cache legal anchors per grid to avoid recomputation
            grid_hash = get_grid_hash(grid)
            if grid_hash not in legal_anchors_cache:
                legal_anchors_cache[grid_hash] = compute_legal_anchors(grid)
            legal_anchors_set = legal_anchors_cache[grid_hash]
            
            # verify expert anchor is legal (should always be true)
            phase0_action = anchor_to_flat_idx(r1, c1)
            if phase0_action not in legal_anchors_set:
                # skip
                print(f"Warning: Expert anchor ({r1},{c1}) has no legal extents")
                continue
            
            phase0_obs = build_observation(grid, phase=0, selected_anchor=None)
            
            # build mask: include all anchors if negative examples enabled, otherwise only legal
            phase0_mask = torch.zeros(170, dtype=torch.bool)
            if include_negative_examples:
                # include all anchors (legal + illegal) so policy can learn to avoid illegal ones
                phase0_mask.fill_(True)
            else:
                # only include legal anchors (old behavior)
                for legal_anchor_idx in sorted(legal_anchors_set):
                    phase0_mask[legal_anchor_idx] = True
            
            phase0_data.append({
                'obs': torch.from_numpy(phase0_obs).float(),
                'action': torch.tensor(phase0_action, dtype=torch.long),
                'mask': phase0_mask,
                'is_positive': True,  # mark as positive example
            })
            
            # generate negative examples for Phase-0 if enabled
            if include_negative_examples:
                illegal_anchors_set = compute_illegal_anchors(grid, legal_anchors_set)
                if illegal_anchors_set:
                    # sample negative anchors according to ratio (probabilistic: generate with probability = ratio)
                    num_negative = 1 if random.random() < negative_example_ratio else 0
                    if num_negative > 0:
                        sampled_illegal = random.sample(list(illegal_anchors_set), min(num_negative, len(illegal_anchors_set)))
                    else:
                        sampled_illegal = []
                    
                    for illegal_anchor_idx in sampled_illegal:
                        phase0_data.append({
                            'obs': torch.from_numpy(phase0_obs).float(),  # same observation
                            'action': torch.tensor(illegal_anchor_idx, dtype=torch.long),  # illegal anchor
                            'mask': phase0_mask,  # all anchors included
                            'is_positive': False,  # mark as negative example
                        })
            
            # phase-1: select extent (r2, c2) given anchor (r1, c1)
            phase1_obs = build_observation(grid, phase=1, selected_anchor=(r1, c1))
            phase1_action_compact = extent_to_flat_idx(r1, c1, r2, c2)
            
            # phase-1: only include legal extents (sum=10), not all geometrically valid ones
            # cache legal extents per (grid, anchor) to avoid recomputation
            phase0_action = anchor_to_flat_idx(r1, c1)
            cache_key = (grid_hash, phase0_action)
            if cache_key not in legal_extents_cache:
                legal_extents_cache[cache_key] = compute_legal_extents(grid, r1, c1)
            legal_extents_set = legal_extents_cache[cache_key]
            
            # periodic cache statistics
            if processed_steps % 10000 == 0:
                print(f"\n  Progress: {processed_steps}/{total_steps} steps | "
                      f"Cache: {len(legal_anchors_cache)} unique grids, "
                      f"{len(legal_extents_cache)} (grid,anchor) pairs | "
                      f"Examples: {len(phase0_data)} Phase-0, {len(phase1_data)} Phase-1")
            
            # verify expert action is legal (should always be true)
            if phase1_action_compact not in legal_extents_set:
                # skip this example if expert action is not legal (shouldn't happen, but handle gracefully)
                print(f"Warning: Expert extent {phase1_action_compact} not in legal set for anchor ({r1},{c1})")
                continue
            
            # build mask: include all geometrically valid extents if negative examples enabled
            max_valid_count = (10 - r1) * (17 - c1)
            phase1_mask = torch.zeros(170, dtype=torch.bool)
            if include_negative_examples:
                # include all geometrically valid extents (legal + illegal) so policy can learn
                for idx in range(min(max_valid_count, 170)):
                    phase1_mask[idx] = True
            else:
                # only include legal extents (old behavior)
                for legal_idx in sorted(legal_extents_set):
                    if legal_idx < 170:  # Safety check
                        phase1_mask[legal_idx] = True
            
            phase1_data.append({
                'obs': torch.from_numpy(phase1_obs).float(),
                'action': torch.tensor(phase1_action_compact, dtype=torch.long),
                'mask': phase1_mask,
                'anchor': torch.tensor(phase0_action, dtype=torch.long),
                'is_positive': True,  # mark as positive example
            })
            
            # generate negative examples for Phase-1 if enabled
            if include_negative_examples:
                illegal_extents_set = compute_illegal_extents(grid, r1, c1, legal_extents_set)
                if illegal_extents_set:
                    # sample negative extents according to ratio (probabilistic: generate with probability = ratio)
                    num_negative = 1 if random.random() < negative_example_ratio else 0
                    if num_negative > 0:
                        sampled_illegal = random.sample(list(illegal_extents_set), min(num_negative, len(illegal_extents_set)))
                    else:
                        sampled_illegal = []
                    
                    for illegal_extent_idx in sampled_illegal:
                        phase1_data.append({
                            'obs': torch.from_numpy(phase1_obs).float(),  # same observation
                            'action': torch.tensor(illegal_extent_idx, dtype=torch.long),  # illegal extent
                            'mask': phase1_mask,  # all extents included
                            'anchor': torch.tensor(phase0_action, dtype=torch.long),
                            'is_positive': False,  # mark as negative example
                        })
    
    total_examples = len(phase0_data) + len(phase1_data)
    phase0_positive = sum(1 for d in phase0_data if d.get('is_positive', True))
    phase0_negative = len(phase0_data) - phase0_positive
    phase1_positive = sum(1 for d in phase1_data if d.get('is_positive', True))
    phase1_negative = len(phase1_data) - phase1_positive
    
    print(f"Processed {len(phase0_data)} Phase-0 examples ({phase0_positive} positive, {phase0_negative} negative)")
    print(f"Processed {len(phase1_data)} Phase-1 examples ({phase1_positive} positive, {phase1_negative} negative)")
    print(f"Total training examples: {total_examples}")
    print(f"Cache statistics:")
    print(f"  Unique grid states (legal anchors cache): {len(legal_anchors_cache)}")
    print(f"  Unique (grid, anchor) pairs (legal extents cache): {len(legal_extents_cache)}")
    print(f"  Cache hit rate: {100 * (1 - len(legal_anchors_cache) / max(total_examples, 1)):.1f}% (anchors), "
          f"{100 * (1 - len(legal_extents_cache) / max(len(phase1_data), 1)):.1f}% (extents)")
    return phase0_data, phase1_data


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
        logits, _ = policy(obs, masks)
        
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


def compute_sft_loss(
    policy: nn.Module,
    obs: torch.Tensor,
    actions: torch.Tensor,
    masks: torch.Tensor,
    is_positive: Optional[torch.Tensor] = None,
    negative_loss_weight: float = 2.0,
) -> Tuple[torch.Tensor, Dict]:
    """Compute SFT loss. Masks can be sparse (only legal actions) or contiguous
    
    For positive examples: standard cross-entropy to maximize probability of correct (legal) action
    For negative examples: penalize high probability on illegal action using -log(1 - prob(illegal))
    This ensures the model learns to avoid illegal actions rather than being encouraged to predict them.
    """
    logits, _ = policy(obs, masks)  # [batch_size, 170]
    
    # compute loss for each sample
    # masks may be sparse (True only at legal extent indices) or contiguous
    losses = []
    correct = 0
    total = 0
    negative_correct = 0  # for negative examples: correct = model does NOT predict illegal action
    negative_total = 0
    
    for b in range(obs.size(0)):
        mask = masks[b]  # [170]
        valid_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)  # [valid_count]
        # ensure valid_indices is 1D (squeeze might make it 0D if single element)
        if valid_indices.dim() == 0:
            valid_indices = valid_indices.unsqueeze(0)
        valid_count = valid_indices.numel()
        
        if valid_count == 0:
            continue
        
        # extract valid logits (only at positions where mask is True)
        valid_logits = logits[b][valid_indices]  # [valid_action_count]
        action = actions[b].item()
        
        # map action index to position in valid_indices
        # action is the original extent index, need to find its position in valid_indices
        action_pos = (valid_indices == action).nonzero(as_tuple=False)
        if action_pos.numel() == 0:
            # action not in valid set (shouldn't happen, but handle gracefully)
            continue
        if action_pos.numel() > 1:
            # multiple matches (shouldn't happen - valid_indices should be unique)
            # take first match
            action_compact = action_pos[0].item()
        else:
            # single match - squeeze to scalar and get item
            action_compact = action_pos.squeeze().item()
        
        # check if this is a negative example (needed before computing loss)
        is_neg = is_positive is not None and not is_positive[b].item() if is_positive is not None else False
        
        # compute loss differently for positive vs negative examples
        if is_neg:
            # for negative examples: penalize high probability on the illegal action
            # use negative log of (1 - prob(illegal)) to push probability down
            log_probs = F.log_softmax(valid_logits, dim=0)
            illegal_log_prob = log_probs[action_compact]
            # loss = -log(1 - exp(illegal_log_prob)) = -log(1 - prob(illegal))
            # use numerical stability: log(1 - exp(x)) = log1p(-exp(x))
            illegal_prob = torch.exp(illegal_log_prob)
            # clamp to avoid numerical issues
            illegal_prob = torch.clamp(illegal_prob, min=1e-8, max=1.0 - 1e-8)
            loss = -torch.log1p(-illegal_prob) * negative_loss_weight
        else:
            # for positive examples: standard cross-entropy to maximize probability of correct action
            loss = F.cross_entropy(valid_logits.unsqueeze(0), torch.tensor([action_compact], device=obs.device))
        losses.append(loss)
        
        # compute accuracy
        pred_action_compact = valid_logits.argmax().item()
        pred_action_original = valid_indices[pred_action_compact].item()
        
        if is_neg:
            # for negative examples: correct = model does NOT predict the illegal action
            negative_total += 1
            if pred_action_original != action:
                negative_correct += 1
        else:
            # for positive examples: correct = model predicts the correct legal action
            total += 1
            if pred_action_original == action:
                correct += 1
    
    if len(losses) == 0:
        # return zero loss if no valid samples
        loss = torch.tensor(0.0, device=obs.device, requires_grad=True)
        accuracy = 0.0
        negative_accuracy = 0.0
    else:
        loss = torch.stack(losses).mean()
        accuracy = correct / total if total > 0 else 0.0
        negative_accuracy = negative_correct / negative_total if negative_total > 0 else 0.0
    
    info = {
        'loss': loss.item(),
        'accuracy': accuracy,
        'negative_accuracy': negative_accuracy,
        'positive_count': total,
        'negative_count': negative_total,
    }
    
    return loss, info


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
        },
        tags=["sft", "fruit-box", "supervised"],
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
    )
    
    # create model
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    print("Model created")
    
    # create optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # training loop
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        
        # combine Phase-0 and Phase-1 data (using ALL datapoints)
        all_data = phase0_data + phase1_data
        random.shuffle(all_data)
        print(f"  Training on {len(all_data)} total examples ({len(phase0_data)} Phase-0 + {len(phase1_data)} Phase-1)")
        
        policy.train()
        epoch_losses = []
        epoch_accuracies = []
        batch_data_for_logging = None
        batch_obs_for_logging = None
        batch_actions_for_logging = None
        batch_masks_for_logging = None
        
        for batch_idx, start in enumerate(tqdm(range(0, len(all_data), config.batch_size), desc="Training")):
            batch_data = all_data[start:start + config.batch_size]
            
            # stack batches (masks already padded to 170)
            batch_obs = torch.stack([d['obs'] for d in batch_data]).to(device)
            batch_actions = torch.stack([d['action'] for d in batch_data]).to(device)
            batch_masks = torch.stack([
                d['mask'] if d['mask'].shape[0] == 170 
                else torch.cat([d['mask'], torch.zeros(170 - d['mask'].shape[0], dtype=torch.bool)])
                for d in batch_data
            ]).to(device)
            batch_is_positive = torch.tensor([d.get('is_positive', True) for d in batch_data], dtype=torch.bool).to(device)
            
            # forward pass
            loss, info = compute_sft_loss(
                policy, batch_obs, batch_actions, batch_masks, batch_is_positive,
                negative_loss_weight=config.negative_loss_weight
            )
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(info)
            epoch_accuracies.append(info['accuracy'])
            
            # save first batch for logging example moves
            if batch_idx == 0:
                batch_data_for_logging = batch_data
                batch_obs_for_logging = batch_obs
                batch_actions_for_logging = batch_actions
                batch_masks_for_logging = batch_masks
        
        # logging
        avg_loss = np.mean([d['loss'] for d in epoch_losses])
        avg_accuracy = np.mean([d['accuracy'] for d in epoch_losses])
        avg_negative_accuracy = np.mean([d.get('negative_accuracy', 0.0) for d in epoch_losses])
        total_positive = sum(d.get('positive_count', 0) for d in epoch_losses)
        total_negative = sum(d.get('negative_count', 0) for d in epoch_losses)
        
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")
        if total_negative > 0:
            print(f"  Positive examples: {total_positive}, Negative examples: {total_negative}")
            print(f"  Negative accuracy (avoiding illegal actions): {avg_negative_accuracy:.4f}")
        
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
        
        log_dict = {
            "epoch": epoch + 1,
            "train/loss": avg_loss,
            "train/accuracy": avg_accuracy,
        }
        if total_negative > 0:
            log_dict["train/negative_accuracy"] = avg_negative_accuracy
            log_dict["train/positive_count"] = total_positive
            log_dict["train/negative_count"] = total_negative
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
    parser.add_argument("--dataset_name", type=str, default="djdumpling/fruit-box")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=20)
    args = parser.parse_args()
    
    config = Config(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
    )
    
    train(config)


if __name__ == "__main__":
    main()