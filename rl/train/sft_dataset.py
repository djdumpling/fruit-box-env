"""Dataset loading and processing for SFT training."""
import random
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

from rl.train.sft_utils import (
    anchor_to_flat_idx,
    extent_to_flat_idx,
    build_observation,
    get_grid_hash,
)
from rl.train.sft_legality import (
    compute_legal_anchors,
    compute_legal_extents,
)


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
                from rl.train.sft_utils import flat_idx_to_extent
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

