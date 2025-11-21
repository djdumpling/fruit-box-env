""" python rl//train/train_sft.py --seed 42 --epochs 200 --batch_size 128 --lr 2e-4 """
# use set-based legality losses
# penalize all illegal actions simulatenously using set-based losses computed from the same forward pass

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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import wandb

from rl.models.policy import CNNPolicy
from fruit_box import Sum10Env


@dataclass
class Config:
    """SFT training configuration"""
    # data
    dataset_name: str = "djdumpling/fruit-box-minimal-area"
    dataset_split: str = "train"
    extra_jsonl: Optional[str] = None
    
    # training
    epochs: int = 200
    batch_size: int = 128  # increased for more stable gradients
    lr: float = 3e-5  # further lowered learning rate for stability (was 5e-5)
    weight_decay: float = 1e-5
    grad_clip_norm: float = 7.0  # increased gradient clipping threshold (was 5.0) to allow larger gradients
    
    # negative examples (for learning legality) - reduced ratio since we use set-based losses
    include_negative_examples: bool = True
    negative_example_ratio: float = 2.0  # reduced from 10.0
    negative_loss_weight: float = 2.0  # target weight after warmup
    negative_loss_weight_start: float = 0.5  # initial weight before schedule
    negative_example_ratio_start: float = 0.25  # gentler initial ratio (was 0.5) for smoother negative introduction
    negative_ratio_warmup_epochs: int = 15  # extended warmup (was 12) for gentler negative introduction
    
    # set-based legality losses (penalize ALL illegal actions simultaneously)
    illegal_mass_alpha: float = 2.0  # target linear penalty on sum of illegal probabilities
    illegal_mass_alpha_start: float = 0.2  # reduced initial alpha for gentler start
    illegal_mass_beta: float = 3.0  # target squared penalty on sum of illegal probabilities (stronger gradients)
    illegal_mass_beta_start: float = 0.5  # reduced initial beta for gentler start
    topk_illegal_k: int = 10  # number of top illegal actions to penalize
    topk_illegal_delta: float = 5.0  # target weight for top-K illegal loss
    topk_illegal_delta_start: float = 0.5  # reduced initial delta for gentler start
    legal_mass_bonus_zeta: float = 0.5  # bonus for high probability on legal actions
    loss_schedule_delay_epochs: int = 5  # delay before ramping loss weights
    loss_schedule_warmup_epochs: int = 20  # extended warmup (was 15) to finish around epoch 25, before curriculum ends at 30
    
    # phase-specific loss weights (Phase-1 has harder task with more illegal extents)
    phase0_loss_weight: float = 1.0  # standard weight for Phase-0 (anchor selection)
    phase1_loss_weight: float = 2.0  # increased weight for Phase-1 (was 1.5) to provide stronger learning signal
    phase1_set_based_multiplier: float = 2.0  # increased multiplier for set-based losses in Phase-1 (was 1.5) to penalize illegal extents more
    
    # auxiliary head warmup
    sum_prediction_loss_weight: float = 0.1  # target weight for sum prediction head
    sum_prediction_loss_start: float = 0.02  # initial weight before warmup
    sum_prediction_loss_warmup_epochs: int = 15  # warmup to delay sum prediction loss (finish at epoch 15, during curriculum)
    
    # curriculum learning
    curriculum_legal_only_epochs: int = 15  # extended legal-only period (was 10) to give model stronger foundation before illegal actions
    use_curriculum: bool = True  # enable curriculum learning
    
    # turn-aware curriculum (filter by turn number and adjust extent limits)
    turn_based_curriculum: bool = True  # enable turn-based filtering and extent limits
    turn_threshold: int = 25  # turn < 25 = early game (more small extents), turn >= 25 = late game (more large extents)
    turn_curriculum_epochs: int = 30  # extended to match extent curriculum (was 20) - epochs to gradually include late-game examples
    turn_early_max_extent_size: int = 6  # max extent size for early-game examples (turn < 25)
    turn_late_max_extent_size: int = 16  # max extent size for late-game examples (turn >= 25)
    
    # extent-size curriculum learning (focus on small extents early)
    extent_curriculum_epochs: int = 30  # extended curriculum (was 25) for smoother transition and better stability
    min_extent_size: int = 2  # minimum (dr, dc) size to include early (e.g., max(dr, dc) >= 2)
    max_extent_size_early: int = 4  # maximum extent size in early curriculum (e.g., max(dr, dc) <= 4)
    extent_curriculum_final_size: int = 16  # target max extent size once curriculum finishes
    extent_curriculum_expansion_rate: float = 0.5  # per-epoch expansion rate for max_extent_size (slower expansion)
    
    # instrumentation / debugging
    instrument_batches: bool = True  # log batch-level stats for early epochs
    instrument_batches_epochs: int = 5  # number of epochs to capture per-batch stats
    instrument_batches_every: int = 10  # log every N batches
    
    # reward-weighted sampling and context-aware loss
    use_reward_weighted_sampling: bool = True  # sample examples with probability proportional to reward^alpha
    reward_sampling_alpha: float = 1.2  # exponent for reward-weighted sampling (higher = more emphasis on high rewards)
    use_context_aware_reward_weighting: bool = True  # weight loss by reward normalized by game state category
    context_aware_early_threshold: float = 0.5  # grid density threshold for early-game (dense) vs late-game (sparse)
    context_aware_trajectory_threshold: int = 30  # step threshold for early-game vs late-game
    
    # other
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 5
    init_checkpoint: Optional[str] = None


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
    """Find all legal extents for a given anchor
    
    Note: extent_idx=0 represents (dr=0, dc=0) which is never legal (single cell can't sum to 10),
    so we skip it explicitly.
    """
    temp_env = Sum10Env()
    temp_env.reset(grid=grid.copy())
    
    legal_extents_set = set()
    max_valid_count = (10 - r1) * (17 - c1)
    for extent_idx in range(max_valid_count):
        # Skip extent_idx=0 (dr=0, dc=0) - single cell can never sum to 10
        if extent_idx == 0:
            continue
        
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
    """Find all geometrically valid extents that DON'T sum to 10
    
    Note: Excludes extent_idx=0 (dr=0, dc=0) since single cell can never sum to 10.
    """
    max_valid_count = (10 - r1) * (17 - c1)
    all_extents = set(range(max_valid_count))
    illegal_extents_set = all_extents - legal_extents_set
    # Remove idx=0 (dr=0, dc=0) - single cell can never sum to 10
    illegal_extents_set.discard(0)
    return illegal_extents_set


def get_pareto_frontier_extents(legal_extents_set: set, r1: int, c1: int) -> List[Tuple[int, int, int]]:
    """Find Pareto frontier: minimal (dr, dc) pairs that are legal
    
    Returns list of (dr, dc, extent_idx) tuples sorted by (dr, dc).
    A pair (dr, dc) is on the frontier if no other legal pair (dr', dc') exists 
    where dr' < dr AND dc' < dc.
    """
    frontier = []
    for extent_idx in legal_extents_set:
        r2, c2 = flat_idx_to_extent(r1, c1, extent_idx)
        dr = r2 - r1
        dc = c2 - c1
        frontier.append((dr, dc, extent_idx))
    
    # Find minimal pairs: (dr, dc) is on frontier if no (dr', dc') with dr' < dr AND dc' < dc
    frontier_minimal = []
    for dr, dc, extent_idx in frontier:
        is_minimal = True
        for dr_other, dc_other, _ in frontier:
            if dr_other < dr and dc_other < dc:
                is_minimal = False
                break
        if is_minimal:
            frontier_minimal.append((dr, dc, extent_idx))
    
    # Sort by (dr, dc) for consistency
    frontier_minimal.sort()
    return frontier_minimal


def get_hard_negatives_near_frontier(
    frontier: List[Tuple[int, int, int]], 
    r1: int, 
    c1: int, 
    max_valid_count: int,
    illegal_extents_set: set
) -> set:
    """Get hard negative extents just beyond or strictly smaller than Pareto frontier
    
    For each frontier extent (dr, dc):
    - Beyond frontier: (dr+1, dc) and (dr, dc+1) - these sum > 10
    - Strictly smaller: (dr-1, dc) if dr > 0, and (dr, dc-1) if dc > 0 - these sum < 10
    
    Returns set of hard negative extent indices.
    
    Note: Includes explicit bounds checking to prevent out-of-bounds extents.
    """
    hard_negatives = set()
    
    for dr, dc, _ in frontier:
        # Beyond frontier: (dr+1, dc) and (dr, dc+1)
        # These are larger in at least one dimension, so sum > 10
        # Check bounds: r1 + dr + 1 < 10 AND c1 + dc < 17
        if dr + 1 < (10 - r1) and c1 + dc < 17:
            try:
                r2_beyond = r1 + dr + 1
                c2_beyond = c1 + dc
                # Double-check bounds
                if 0 <= r2_beyond < 10 and 0 <= c2_beyond < 17:
                    idx_beyond_r = extent_to_flat_idx(r1, c1, r2_beyond, c2_beyond)
                    if idx_beyond_r < max_valid_count and idx_beyond_r in illegal_extents_set:
                        hard_negatives.add(idx_beyond_r)
            except (ValueError, IndexError):
                pass  # Invalid extent, skip
        
        # Check bounds: r1 + dr < 10 AND c1 + dc + 1 < 17
        if r1 + dr < 10 and dc + 1 < (17 - c1):
            try:
                r2_beyond = r1 + dr
                c2_beyond = c1 + dc + 1
                # Double-check bounds
                if 0 <= r2_beyond < 10 and 0 <= c2_beyond < 17:
                    idx_beyond_c = extent_to_flat_idx(r1, c1, r2_beyond, c2_beyond)
                    if idx_beyond_c < max_valid_count and idx_beyond_c in illegal_extents_set:
                        hard_negatives.add(idx_beyond_c)
            except (ValueError, IndexError):
                pass  # Invalid extent, skip
        
        # Strictly smaller: (dr-1, dc) and (dr, dc-1)
        # These are smaller in at least one dimension, so sum < 10
        # Check bounds: dr > 0 AND r1 + dr - 1 < 10 AND c1 + dc < 17
        if dr > 0 and r1 + dr - 1 < 10 and c1 + dc < 17:
            try:
                r2_smaller = r1 + dr - 1
                c2_smaller = c1 + dc
                # Double-check bounds
                if 0 <= r2_smaller < 10 and 0 <= c2_smaller < 17:
                    idx_smaller_r = extent_to_flat_idx(r1, c1, r2_smaller, c2_smaller)
                    if idx_smaller_r < max_valid_count and idx_smaller_r in illegal_extents_set:
                        hard_negatives.add(idx_smaller_r)
            except (ValueError, IndexError):
                pass  # Invalid extent, skip
        
        # Check bounds: dc > 0 AND r1 + dr < 10 AND c1 + dc - 1 < 17
        if dc > 0 and r1 + dr < 10 and c1 + dc - 1 < 17:
            try:
                r2_smaller = r1 + dr
                c2_smaller = c1 + dc - 1
                # Double-check bounds
                if 0 <= r2_smaller < 10 and 0 <= c2_smaller < 17:
                    idx_smaller_c = extent_to_flat_idx(r1, c1, r2_smaller, c2_smaller)
                    if idx_smaller_c < max_valid_count and idx_smaller_c in illegal_extents_set:
                        hard_negatives.add(idx_smaller_c)
            except (ValueError, IndexError):
                pass  # Invalid extent, skip
    
    return hard_negatives


def load_and_process_dataset(
    dataset_name: str,
    dataset_split: str,
    seed: Optional[int] = None,
    include_negative_examples: bool = True,
    negative_example_ratio: float = 0.5,
    extra_jsonl: Optional[str] = None,
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
    
    if extra_jsonl:
        print(f"Loading extra corrective dataset from {extra_jsonl}...")
        extra_ds = load_dataset(
            "json",
            data_files={"train": extra_jsonl},
            split="train",
        )
        hf_dataset = concatenate_datasets([hf_dataset, extra_ds])
        print(f"Combined dataset size: {len(hf_dataset)} rows")
    
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
            
            # compute reward, step number, and grid density for context-aware weighting
            reward = step.get("reward", 0)  # cells cleared by this move
            step_num = step.get("step", 0)  # step number in trajectory
            grid_density = (grid > 0).sum() / (10 * 17)  # fraction of non-zero cells (0.0 to 1.0)
            
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
            
            # store positive example with metadata for on-the-fly negative generation
            phase0_data.append({
                'obs': torch.from_numpy(phase0_obs).float(),
                'action': torch.tensor(phase0_action, dtype=torch.long),
                'mask': phase0_mask,
                'is_positive': True,
                'grid': grid.copy(),  # needed for negative generation
                'legal_anchors_set': legal_anchors_set.copy(),  # needed for negative generation
                'phase': 0,
                'reward': reward,  # for reward-weighted sampling and loss
                'step_num': step_num,  # for trajectory position weighting
                'grid_density': grid_density,  # for context-aware reward weighting
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
            # Also check that it's not (0,0) - single cell can never sum to 10
            if phase1_action_compact == 0:
                print(f"Warning: Expert extent is (0,0) for anchor ({r1},{c1}) - skipping (single cell can't sum to 10)")
                continue
            if phase1_action_compact not in legal_extents_set:
                # skip this example if expert action is not legal (shouldn't happen, but handle gracefully)
                print(f"Warning: Expert extent {phase1_action_compact} not in legal set for anchor ({r1},{c1})")
                continue
            
            # build mask: include all geometrically valid extents if negative examples enabled
            max_valid_count = (10 - r1) * (17 - c1)
            phase1_mask = torch.zeros(170, dtype=torch.bool)
            if include_negative_examples:
                # include all geometrically valid extents (legal + illegal) so policy can learn
                # Skip idx=0 (dr=0, dc=0) - single cell can never sum to 10
                for idx in range(1, min(max_valid_count, 170)):
                    phase1_mask[idx] = True
            else:
                # only include legal extents (old behavior)
                # legal_extents_set already excludes idx=0 from compute_legal_extents
                for legal_idx in sorted(legal_extents_set):
                    if legal_idx < 170 and legal_idx > 0:  # Safety check: skip idx=0
                        phase1_mask[legal_idx] = True
            
            # store positive example with metadata for on-the-fly negative generation
            phase1_data.append({
                'obs': torch.from_numpy(phase1_obs).float(),
                'action': torch.tensor(phase1_action_compact, dtype=torch.long),
                'mask': phase1_mask,
                'anchor': torch.tensor(phase0_action, dtype=torch.long),
                'is_positive': True,
                'grid': grid.copy(),  # needed for negative generation
                'r1': r1,  # needed for negative generation
                'c1': c1,  # needed for negative generation
                'legal_extents_set': legal_extents_set.copy(),  # needed for negative generation
                'phase': 1,
                'reward': reward,  # for reward-weighted sampling and loss
                'step_num': step_num,  # for trajectory position weighting
                'grid_density': grid_density,  # for context-aware reward weighting
            })
    
    # only positive examples are stored (negatives generated on-the-fly)
    total_positive = len(phase0_data) + len(phase1_data)
    print(f"Processed {len(phase0_data)} Phase-0 positive examples")
    print(f"Processed {len(phase1_data)} Phase-1 positive examples")
    if include_negative_examples:
        print(f"Negatives will be generated on-the-fly during training with ratio {negative_example_ratio}:1")
        print(f"  (Effective training examples per epoch: ~{int(total_positive * (1 + negative_example_ratio))})")
    print(f"Cache statistics:")
    print(f"  Unique grid states (legal anchors cache): {len(legal_anchors_cache)}")
    print(f"  Unique (grid, anchor) pairs (legal extents cache): {len(legal_extents_cache)}")
    print(f"  Cache hit rate: {100 * (1 - len(legal_anchors_cache) / max(total_positive, 1)):.1f}% (anchors), "
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


def generate_negatives_for_positive(
    positive_example: Dict,
    negative_example_ratio: float,
) -> Tuple[List[Dict], Dict]:
    """Generate negative examples for a positive example on-the-fly
    
    Returns:
        negatives: List of negative examples
        stats: Dict with 'used_hard_negatives' (bool) and 'num_hard_negatives' (int)
    """
    negatives = []
    stats = {'used_hard_negatives': False, 'num_hard_negatives': 0}
    grid = positive_example['grid']
    phase = positive_example['phase']
    
    if phase == 0:
        # Phase-0: generate negative anchors
        legal_anchors_set = positive_example['legal_anchors_set']
        illegal_anchors_set = compute_illegal_anchors(grid, legal_anchors_set)
        if illegal_anchors_set:
            base_count = int(negative_example_ratio)
            fractional = negative_example_ratio - base_count
            num_negative = base_count + (1 if random.random() < fractional else 0)
            if num_negative > 0:
                sampled_illegal = random.sample(
                    list(illegal_anchors_set), 
                    min(num_negative, len(illegal_anchors_set))
                )
                for illegal_anchor_idx in sampled_illegal:
                    negatives.append({
                        'obs': positive_example['obs'].clone(),
                        'action': torch.tensor(illegal_anchor_idx, dtype=torch.long),
                        'mask': positive_example['mask'].clone(),
                        'is_positive': False,
                        'legal_anchors_set': legal_anchors_set.copy(),  # needed for set-based losses
                        'phase': 0,
                    })
    else:
        # Phase-1: generate negative extents using Pareto frontier hard negative mining
        r1 = positive_example['r1']
        c1 = positive_example['c1']
        legal_extents_set = positive_example['legal_extents_set']
        illegal_extents_set = compute_illegal_extents(grid, r1, c1, legal_extents_set)
        if illegal_extents_set:
            base_count = int(negative_example_ratio)
            fractional = negative_example_ratio - base_count
            num_negative = base_count + (1 if random.random() < fractional else 0)
            if num_negative > 0:
                # Try to use hard negatives from Pareto frontier
                max_valid_count = (10 - r1) * (17 - c1)
                hard_negatives = set()
                
                if legal_extents_set:
                    # Compute Pareto frontier
                    frontier = get_pareto_frontier_extents(legal_extents_set, r1, c1)
                    if frontier:
                        # Get hard negatives near frontier
                        hard_negatives = get_hard_negatives_near_frontier(
                            frontier, r1, c1, max_valid_count, illegal_extents_set
                        )
                
                # Use hard negatives if available, otherwise fallback to random
                if hard_negatives and len(hard_negatives) > 0:
                    stats['used_hard_negatives'] = True
                    stats['num_hard_negatives'] = len(hard_negatives)
                    sampled_illegal = random.sample(
                        list(hard_negatives),
                        min(num_negative, len(hard_negatives))
                    )
                else:
                    # Fallback to random sampling
                    sampled_illegal = random.sample(
                        list(illegal_extents_set),
                        min(num_negative, len(illegal_extents_set))
                    )
                
                for illegal_extent_idx in sampled_illegal:
                    # Safety checks: exclude idx=0 and ensure within bounds
                    if illegal_extent_idx < 170 and illegal_extent_idx > 0:
                        negatives.append({
                            'obs': positive_example['obs'].clone(),
                            'action': torch.tensor(illegal_extent_idx, dtype=torch.long),
                            'mask': positive_example['mask'].clone(),
                            'anchor': positive_example['anchor'].clone(),
                            'is_positive': False,
                            'legal_extents_set': legal_extents_set.copy(),  # needed for set-based losses
                            'phase': 1,
                        })
    
    return negatives, stats


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
    
    # create model
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    if config.init_checkpoint:
        state_dict = torch.load(config.init_checkpoint, map_location=device)
        policy.load_state_dict(state_dict)
        print(f"Model initialized from checkpoint: {config.init_checkpoint}")
    else:
        print("Model created from scratch")
    
    # create optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # training loop
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        
        # curriculum learning: use legal-only masks for first N epochs
        use_legal_only_masks = (config.use_curriculum and 
                               epoch < config.curriculum_legal_only_epochs)
        if use_legal_only_masks:
            print(f"  Curriculum: Using legal-only masks (epoch {epoch + 1} < {config.curriculum_legal_only_epochs})")
        else:
            print(f"  Curriculum: Using all-geometric masks with set-based losses")
        
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
        
        if config.extent_curriculum_epochs <= 0:
            extent_curriculum_progress = 1.0
        else:
            extent_curriculum_progress = min(
                1.0,
                (epoch + 1) / max(config.extent_curriculum_epochs, 1),
            )
        current_max_extent_size = int(round(interp(
            config.max_extent_size_early,
            config.extent_curriculum_final_size,
            extent_curriculum_progress,
        )))
        current_max_extent_size = max(current_max_extent_size, config.max_extent_size_early)
        
        # Gradual illegal exposure: start at 0 during legal-only period, then ramp up
        if use_legal_only_masks:
            # During legal-only period: no negative examples
            current_negative_ratio = 0.0
        else:
            # After legal-only period: gradually increase negative ratio
            # Compute progress from end of legal-only period
            epochs_since_legal_only = max(0, epoch - config.curriculum_legal_only_epochs + 1)
            if config.negative_ratio_warmup_epochs <= 0:
                negative_ratio_progress = 1.0
            else:
                negative_ratio_progress = min(
                    1.0,
                    epochs_since_legal_only / max(config.negative_ratio_warmup_epochs, 1),
                )
            current_negative_ratio = interp(
                config.negative_example_ratio_start,
                config.negative_example_ratio,
                negative_ratio_progress,
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
        if not use_legal_only_masks:
            print(f"  Set-based weights this epoch → alpha={current_illegal_mass_alpha:.2f}, "
                  f"beta={current_illegal_mass_beta:.2f}, topk_delta={current_topk_illegal_delta:.2f}")
        print(f"  Sum-head loss weight={current_sum_pred_loss_weight:.3f} (progress={sum_loss_progress:.2f})")
        
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
            
            # update masks based on curriculum learning
            if use_legal_only_masks:
                # curriculum phase: use legal-only masks (reconstruct from legal sets)
                updated_masks = []
                for i, d in enumerate(batch_data):
                    mask = torch.zeros(170, dtype=torch.bool)
                    legal_set = legal_actions_sets[i]
                    for legal_idx in legal_set:
                        if legal_idx < 170 and legal_idx > 0:  # Skip idx=0 (dr=0, dc=0)
                            mask[legal_idx] = True
                    updated_masks.append(mask)
                batch_masks = torch.stack(updated_masks).to(device)
            else:
                # full phase: use all-geometric masks (already in batch_data)
                batch_masks = torch.stack([
                    d["mask"] if d["mask"].shape[0] == 170 
                    else torch.cat([d["mask"], torch.zeros(170 - d["mask"].shape[0], dtype=torch.bool)])
                    for d in batch_data
                ]).to(device)
            
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
            use_set_based = not use_legal_only_masks
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
        if not use_legal_only_masks:
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
        if not use_legal_only_masks:
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