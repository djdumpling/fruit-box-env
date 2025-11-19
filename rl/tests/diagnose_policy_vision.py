#!/usr/bin/env python3
"""
Diagnostic script to check if the policy can see grid values and learn to sum rectangles.

This script will:
1. Show what the policy observation looks like
2. Test if the policy can distinguish between rectangles with different sums
3. Visualize the observation channels
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from rl.models.policy import CNNPolicy
from rl.envs.sum10_env import Sum10GymEnv
from rl.envs.split_wrapper import TwoPhaseWrapper
from fruit_box import Sum10Env


def print_observation_channels(obs: np.ndarray, title: str = "Observation"):
    """Print all 4 channels of the observation."""
    print(f"\n{title}:")
    print("=" * 80)
    
    # Channel 0: Normalized values
    print("\nChannel 0: Normalized Grid Values (0-9 → 0-1)")
    print("   " + " ".join([f"{i:2d}" for i in range(17)]))
    for r in range(10):
        row = " ".join([f"{obs[0, r, c]:.2f}" for c in range(17)])
        print(f"{r:2d} {row}")
    
    # Channel 1: Nonzero mask
    print("\nChannel 1: Nonzero Mask (1 if cell > 0, else 0)")
    print("   " + " ".join([f"{i:2d}" for i in range(17)]))
    for r in range(10):
        row = " ".join([f"{int(obs[1, r, c]):2d}" for c in range(17)])
        print(f"{r:2d} {row}")
    
    # Channel 2: Anchor mask
    print("\nChannel 2: Anchor Mask (1 at selected anchor, else 0)")
    print("   " + " ".join([f"{i:2d}" for i in range(17)]))
    for r in range(10):
        row = " ".join([f"{int(obs[2, r, c]):2d}" for c in range(17)])
        print(f"{r:2d} {row}")
    
    # Channel 3: Phase mask
    print("\nChannel 3: Phase Mask (0 = Phase-0, 1 = Phase-1)")
    print("   " + " ".join([f"{i:2d}" for i in range(17)]))
    for r in range(10):
        row = " ".join([f"{int(obs[3, r, c]):2d}" for c in range(17)])
        print(f"{r:2d} {row}")


def test_rectangle_sum_visibility():
    """Test if the policy can see values within a rectangle."""
    print("=" * 80)
    print("TEST: Can the policy see grid values within a rectangle?")
    print("=" * 80)
    
    # Create a simple test grid
    grid = np.zeros((10, 17), dtype=np.uint8)
    # Create a rectangle that sums to 10: (0,0) to (1,1) = [5, 2, 1, 2] = 10
    grid[0, 0] = 5
    grid[0, 1] = 2
    grid[1, 0] = 1
    grid[1, 1] = 2
    
    # Create another rectangle that sums to 8: (0,2) to (0,3) = [3, 5] = 8
    grid[0, 2] = 3
    grid[0, 3] = 5
    
    print("\nTest Grid:")
    print("   " + " ".join([f"{i:2d}" for i in range(17)]))
    for r in range(10):
        row = " ".join([f"{int(grid[r, c]):2d}" for c in range(17)])
        print(f"{r:2d} {row}")
    
    # Create environment and get Phase-1 observation
    env = Sum10GymEnv(initial_grid=grid.copy())
    wrapped_env = TwoPhaseWrapper(env, curriculum_legal_only=False, curriculum_updates=0)
    
    # Select anchor (0, 0)
    obs, _ = wrapped_env.reset()
    anchor_idx = 0  # (0, 0)
    obs_after_anchor, _, _, _, _ = wrapped_env.step(anchor_idx)
    
    # Convert to numpy for visualization
    obs_np = obs_after_anchor.squeeze(0).numpy()
    
    print_observation_channels(obs_np, "Phase-1 Observation (after selecting anchor (0,0))")
    
    # Check what rectangles are available
    validation_env = Sum10Env()
    validation_env.reset(grid=grid.copy())
    
    print("\n" + "=" * 80)
    print("Available rectangles from anchor (0,0):")
    print("=" * 80)
    
    r1, c1 = 0, 0
    width = 17 - c1
    for dr in range(2):  # Show first few rows
        for dc in range(4):  # Show first few columns
            r2 = r1 + dr
            c2 = c1 + dc
            extent_idx = dr * width + dc
            rect_sum = validation_env.box_sum(r1, c1, r2, c2)
            area = (r2 - r1 + 1) * (c2 - c1 + 1)
            
            # Extract values in rectangle
            rect_values = []
            for rr in range(r1, r2 + 1):
                for cc in range(c1, c2 + 1):
                    rect_values.append(int(grid[rr, cc]))
            
            status = "✓ LEGAL" if rect_sum == 10 else "✗ ILLEGAL"
            print(f"  Extent idx {extent_idx:2d}: ({r1},{c1}) -> ({r2},{c2}) | "
                  f"Area: {area} | Values: {rect_values} | Sum: {rect_sum:2d} [{status}]")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHT:")
    print("=" * 80)
    print("The policy CAN see grid values (Channel 0), but it needs to learn to:")
    print("1. Look at the anchor position (Channel 2 shows this)")
    print("2. For each potential extent (r2, c2), mentally 'sum' the values in the rectangle")
    print("3. Select the extent where the sum equals 10")
    print("\nThis is a complex spatial reasoning task that requires the CNN to:")
    print("- Understand the relationship between anchor and extent positions")
    print("- Learn to sum values within arbitrary rectangles")
    print("- Generalize this to unseen grid patterns")
    print("\nThe fact that the model selects extent_idx=0 (1x1 rectangle) suggests")
    print("it's NOT learning to properly sum rectangles - it's just defaulting to")
    print("the smallest possible rectangle.")


if __name__ == "__main__":
    test_rectangle_sum_visibility()

