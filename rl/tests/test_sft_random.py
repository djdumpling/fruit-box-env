#!/usr/bin/env python3
"""
Evaluate the SFT policy on randomly generated Sum-10 grids.

Unlike test_sft.py (which replays dataset grids), this script fabricates new
boards by planting at least one legal rectangle per grid, then reuses the
standard evaluation loop from test_sft.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.tests.test_sft import test_policy_with_all_masks  # noqa: E402
from rl.tests.test_sft import get_grid_hash  # reuse helper for consistency
from fruit_box import Sum10Env  # noqa: E402


def plant_legal_rectangle(grid: np.ndarray, rng: np.random.Generator) -> bool:
    """Insert at least one rectangle whose values sum to 10.

    Returns True on success, False if planting failed after several attempts.
    """
    for _ in range(200):
        r1 = rng.integers(0, 10)
        c1 = rng.integers(0, 17)
        max_r2 = min(r1 + rng.integers(1, 4), 9)
        max_c2 = min(c1 + rng.integers(1, 4), 16)
        if max_r2 <= r1 or max_c2 <= c1:
            continue
        r2 = rng.integers(r1 + 1, max_r2 + 1)
        c2 = rng.integers(c1 + 1, max_c2 + 1)
        area = (r2 - r1 + 1) * (c2 - c1 + 1)
        if area <= 1:
            continue  # need at least 2 cells to make sum=10 with digits 0-9
        values = rng.integers(0, 10, size=area - 1)
        needed = 10 - int(values.sum())
        if 0 <= needed <= 9:
            # fill rectangle row-major
            idx = 0
            for rr in range(r1, r2 + 1):
                for cc in range(c1, c2 + 1):
                    if idx < len(values):
                        grid[rr, cc] = values[idx]
                        idx += 1
                    else:
                        grid[rr, cc] = needed
            return True
    return False


def generate_random_grid(seed: int) -> np.ndarray:
    """Generate a random grid with at least one legal move."""
    rng = np.random.default_rng(seed)
    grid = rng.integers(0, 10, size=(10, 17), dtype=np.uint8)
    if not plant_legal_rectangle(grid, rng):
        # Fallback: create a simple 2x2 rectangle manually summing to 10
        grid.fill(0)
        grid[0, 0] = 5
        grid[0, 1] = 2
        grid[1, 0] = 1
        grid[1, 1] = 2  # 5+2+1+2 = 10
    return grid


def main():
    parser = argparse.ArgumentParser(description="Test SFT policy on random grids.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SFT checkpoint")
    parser.add_argument("--num_grids", type=int, default=25, help="Number of random grids to test")
    parser.add_argument("--max_moves", type=int, default=60, help="Maximum moves per grid")
    parser.add_argument("--seed", type=int, default=12345, help="Base RNG seed for grid generation")
    parser.add_argument("--collect_examples", action="store_true", help="Emit corrective examples")
    parser.add_argument("--output_examples", type=str, default=None, help="Path for corrective JSONL")
    parser.add_argument("--verbose", action="store_true", help="Print per-grid stats")
    args = parser.parse_args()

    # Generate unique random grids
    grids = []
    seen_hashes = set()
    seed = args.seed
    while len(grids) < args.num_grids:
        grid = generate_random_grid(seed)
        seed += 1
        sig = get_grid_hash(grid)
        if sig in seen_hashes:
            continue
        seen_hashes.add(sig)
        grids.append(grid)

    print(f"Generated {len(grids)} random grids (seed base {args.seed}).")

    test_policy_with_all_masks(
        checkpoint_path=args.checkpoint,
        grids=grids,
        max_moves_per_grid=args.max_moves,
        verbose=args.verbose,
        collect_examples=args.collect_examples,
        output_examples_path=args.output_examples,
    )


if __name__ == "__main__":
    main()

