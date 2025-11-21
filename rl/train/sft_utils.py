"""Utility functions for SFT training: action space conversions and observation building."""
import numpy as np
from typing import Tuple, Optional


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

