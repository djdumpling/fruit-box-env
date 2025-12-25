"""Wrapper around Sum10Env for RL training."""
import numpy as np
from typing import Optional, Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces

from fruit_box import Sum10Env


class Sum10GymEnv(gym.Env):
    """Gymnasium wrapper for Sum10Env."""
    
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 4}
    
    def __init__(
        self,
        initial_grid: Optional[np.ndarray] = None,
        max_steps: int = 85,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._initial_grid = initial_grid.copy() if initial_grid is not None else None
        self._default_seed = seed
        self._rng = np.random.default_rng(seed) if seed is not None else None
        self.game_env = Sum10Env()
        
        # observation space: 10x17 grid with values 0-9
        self.observation_space = spaces.Box(low=0.0, high=9.0, shape=(10, 17), dtype=np.float32)
        
        # action space (r1: 0-9, c1: 0-16, r2: 0-9, c2: 0-16)
        self.action_space = spaces.MultiDiscrete([10, 17, 10, 17])
        
        self.total_steps = 0
        self.episode_info = {}
    
    def get_legal_actions(self) -> list:
        """Return list of legal actions as tuples (r1, c1, r2, c2)."""
        legal_moves = self.game_env.enumerate_legal()
        return [(r1, c1, r2, c2) for ((r1, c1, r2, c2), _) in legal_moves]
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        # use provided seed, or fall back to default seed for this environment
        reset_seed = seed if seed is not None else self._default_seed
        super().reset(seed=reset_seed)
        
        # get initial grid from options or use stored/default
        initial_grid = None
        if options and "grid" in options:
            initial_grid = np.array(options["grid"], dtype=np.uint8)
        elif self._initial_grid is not None:
            initial_grid = self._initial_grid.copy()
        
        # if no grid provided, generate a random valid grid
        if initial_grid is None:
            # use environment's persistent RNG if no seed was explicitly provided, otherwise create new one
            if seed is None and self._rng is not None:
                rng = self._rng
            else:
                rng = np.random.default_rng(reset_seed)
            initial_grid = rng.integers(1, 10, size=(10, 17), dtype=np.uint8)
            # ensure sum is divisible by 10 (with max iterations to avoid infinite loop)
            max_iter = 1000
            iter_count = 0
            while initial_grid.sum() % 10 != 0 and iter_count < max_iter:
                initial_grid = rng.integers(1, 10, size=(10, 17), dtype=np.uint8)
                iter_count += 1
        
        # reset env
        self.game_env.reset(grid=initial_grid)
        self.total_steps = 0
        
        observation = self.game_env.grid.copy().astype(np.float32)
        
        info = {
            "turn": 0,
            "legal_moves_count": len(self.game_env.enumerate_legal()),
        }
        self.episode_info = info
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # extract action
        r1, c1, r2, c2 = int(action[0]), int(action[1]), int(action[2]), int(action[3])
        
        step_info = self.game_env.step(r1, c1, r2, c2)
        observation = self.game_env.grid.copy().astype(np.float32)
        
        self.total_steps += 1
        
        if step_info.valid:
            reward = float(step_info.reward)
            terminated = step_info.done
            truncated = self.total_steps >= self.max_steps
            
            info = {
                "valid": True,
                "turn": self.game_env.turn,
                "cells_cleared": step_info.reward,
                "legal_moves_count": len(self.game_env.enumerate_legal()) if not terminated else 0,
                "termination_reason": "no_legal_moves" if terminated else None,
                "truncation_reason": "max_steps" if truncated else None,
            }
        else:
            # invalid move - end episode
            reward = 0.0
            terminated = True
            truncated = False
            
            info = {
                "valid": False,
                "turn": self.game_env.turn,
                "cells_cleared": 0,
                "actual_sum": step_info.sum,
                "legal_moves_count": len(self.game_env.enumerate_legal()),
                "termination_reason": "invalid_move",
            }
        
        self.episode_info = info
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        
        grid = self.game_env.grid
        
        if self.render_mode == "human":
            print("\n" + "=" * 50)
            print(f"Turn: {self.game_env.turn}")
            print("=" * 50)
            for row in grid:
                print(" ".join(f"{cell:2d}" for cell in row))
            print("=" * 50 + "\n")
            return None
        
        elif self.render_mode == "ansi":
            lines = ["=" * 50]
            lines.append(f"Turn: {self.game_env.turn}")
            lines.append("=" * 50)
            for row in grid:
                lines.append(" ".join(f"{cell:2d}" for cell in row))
            lines.append("=" * 50)
            return "\n".join(lines)
        
        elif self.render_mode == "rgb_array":
            img = (grid.astype(np.float32) / 9.0 * 255).astype(np.uint8)
            img_rgb = np.stack([img, img, img], axis=-1)
            return img_rgb
        
        return None

