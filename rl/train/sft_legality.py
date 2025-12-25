"""Legality computation functions for SFT training."""
import numpy as np
from typing import Set
from fruit_box import Sum10Env

from rl.train.sft_utils import anchor_to_flat_idx, flat_idx_to_extent


def compute_legal_anchors(grid: np.ndarray) -> Set[int]:
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


def compute_legal_extents(grid: np.ndarray, r1: int, c1: int) -> Set[int]:
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


def compute_illegal_anchors(grid: np.ndarray, legal_anchors_set: Set[int]) -> Set[int]:
    """Find all anchors that DON'T have any legal extents"""
    all_anchors = set(range(170))
    illegal_anchors_set = all_anchors - legal_anchors_set
    return illegal_anchors_set


def compute_illegal_extents(grid: np.ndarray, r1: int, c1: int, legal_extents_set: Set[int]) -> Set[int]:
    """Find all geometrically valid extents that DON'T sum to 10
    
    Note: Excludes extent_idx=0 (dr=0, dc=0) since single cell can never sum to 10.
    """
    max_valid_count = (10 - r1) * (17 - c1)
    all_extents = set(range(max_valid_count))
    illegal_extents_set = all_extents - legal_extents_set
    # Remove idx=0 (dr=0, dc=0) - single cell can never sum to 10
    illegal_extents_set.discard(0)
    return illegal_extents_set

