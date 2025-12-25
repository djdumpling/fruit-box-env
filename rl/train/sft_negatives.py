"""Hard negative mining and negative example generation for SFT training."""
import random
from typing import Dict, List, Tuple, Set
import torch

from rl.train.sft_utils import extent_to_flat_idx, flat_idx_to_extent
from rl.train.sft_legality import compute_illegal_anchors, compute_illegal_extents


def get_pareto_frontier_extents(legal_extents_set: Set[int], r1: int, c1: int) -> List[Tuple[int, int, int]]:
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
    illegal_extents_set: Set[int]
) -> Set[int]:
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

