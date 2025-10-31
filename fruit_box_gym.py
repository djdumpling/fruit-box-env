import numpy as np
from typing import Dict, Optional, Tuple, Any
import gymnasium as gym
from gymnasium import spaces

from fruit_box import Sum10Env, StepInfo


class FruitBoxGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 4}

    def __init__(
        self,
        initial_grid: Optional[np.ndarray] = None,
        invalid_move_penalty: float = -0.1,
        max_invalid_moves: int = 10,
        terminate_on_invalid: bool = False,
        max_steps: int = 85,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.invalid_move_penalty = invalid_move_penalty
        self.max_invalid_moves = max_invalid_moves
        self.terminate_on_invalid = terminate_on_invalid
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # initial grid for reset
        self._initial_grid = initial_grid.copy() if initial_grid is not None else None
        self.game_env = Sum10Env()
        
        # observation space: 10x17 grid with values 0-9
        self.observation_space = spaces.Box(low=0.0, high=9.0, shape=(10, 17), dtype=np.float32)
        
        # action space (r1: 0-9, c1: 0-16, r2: 0-9, c2: 0-16)
        self.action_space = spaces.MultiDiscrete([10, 17, 10, 17])
        
        # episode tracking
        self.total_reward = 0.0
        self.consecutive_invalid_moves = 0
        self.turn_count = 0
        self.total_steps = 0
        self.episode_info = {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # get initial grid from options or use stored/default
        initial_grid = None
        if options and "grid" in options:
            initial_grid = np.array(options["grid"], dtype=np.uint8)
        elif self._initial_grid is not None:
            initial_grid = self._initial_grid.copy()
        
        # reset env, rewards, etc.
        self.game_env.reset(grid=initial_grid)
        self.total_reward = 0.0
        self.consecutive_invalid_moves = 0
        self.turn_count = 0
        self.total_steps = 0
        
        observation = self.game_env.grid.copy().astype(np.float32)
        
        # prep info dict
        info = {
            "turn": 0,
            "total_reward": 0.0,
            "legal_moves_count": len(self.game_env.enumerate_legal()),
        }
        self.episode_info = info
        
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # validation
        invalid = False
        try:
            a = np.asarray(action).astype(int)
            if a.shape != (4,):
                invalid = True
            else:
                r1, c1, r2, c2 = int(a[0]), int(a[1]), int(a[2]), int(a[3])
        except Exception:
            invalid = True

        if invalid:
            # invalid move
            observation = self.game_env.grid.copy().astype(np.float32)
            reward = self.invalid_move_penalty
            self.total_reward += reward
            self.consecutive_invalid_moves += 1
            self.total_steps += 1
            terminated = False

            if self.terminate_on_invalid or self.consecutive_invalid_moves >= self.max_invalid_moves:
                terminated = True

            if self.total_steps >= self.max_steps:
                truncated = True
                
            info = {
                "valid": False,
                "turn": self.turn_count,
                "total_reward": self.total_reward,
                "cells_cleared": 0,
                "actual_sum": None,
                "consecutive_invalid_moves": self.consecutive_invalid_moves,
                "legal_moves_count": len(self.game_env.enumerate_legal()),
                "invalid_action_format": True,
                "termination_reason": "invalid_move" if terminated else None,
                "truncation_reason": "max_steps" if truncated else None,
            }
            self.episode_info = info
            return observation, reward, terminated, truncated, info
        
        step_info: StepInfo = self.game_env.step(r1, c1, r2, c2)
        observation = self.game_env.grid.copy().astype(np.float32)
        
        # reward and termination
        self.total_steps += 1
        if step_info.valid:
            reward = float(step_info.reward)
            self.total_reward += reward
            self.consecutive_invalid_moves = 0
            self.turn_count += 1
            terminated = step_info.done
            truncated = False

            if self.total_steps >= self.max_steps:
                truncated = True
            
            info = {
                "valid": True,
                "turn": self.turn_count,
                "total_reward": self.total_reward,
                "cells_cleared": step_info.reward,
                "legal_moves_count": len(self.game_env.enumerate_legal()) if not terminated else 0,
                "termination_reason": "no_legal_moves" if terminated else None,
                "truncation_reason": "max_steps" if truncated else None,
            }
        else:
            # invalid
            reward = self.invalid_move_penalty
            self.total_reward += reward
            self.consecutive_invalid_moves += 1
            terminated = False
            truncated = False
            
            # termination due to too many invalid moves
            if self.terminate_on_invalid or self.consecutive_invalid_moves >= self.max_invalid_moves:
                terminated = True
            if self.total_steps >= self.max_steps:
                truncated = True
            
            info = {
                "valid": False,
                "turn": self.turn_count,
                "total_reward": self.total_reward,
                "cells_cleared": 0,
                "actual_sum": step_info.sum,
                "consecutive_invalid_moves": self.consecutive_invalid_moves,
                "legal_moves_count": len(self.game_env.enumerate_legal()),
                "termination_reason": "invalid_move" if terminated else None,
                "truncation_reason": "max_steps" if truncated else None,
            }
        
        self.episode_info = info
        
        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        
        grid = self.game_env.grid
        
        if self.render_mode == "human":
            # print to console
            print("\n" + "=" * 50)
            print(f"Turn: {self.turn_count} | Total Reward: {self.total_reward:.1f}")
            print("=" * 50)
            for row in grid:
                print(" ".join(f"{cell:2d}" for cell in row))
            print("=" * 50 + "\n")
            return None
        
        elif self.render_mode == "ansi":
            # ANSI string representation
            lines = ["=" * 50]
            lines.append(f"Turn: {self.turn_count} | Total Reward: {self.total_reward:.1f}")
            lines.append("=" * 50)
            for row in grid:
                lines.append(" ".join(f"{cell:2d}" for cell in row))
            lines.append("=" * 50)
            return "\n".join(lines)
        
        elif self.render_mode == "rgb_array":
            # grayscale grid
            # normalize to 0-255 range
            img = (grid.astype(np.float32) / 9.0 * 255).astype(np.uint8)
            # Expand to RGB
            img_rgb = np.stack([img, img, img], axis=-1)
            return img_rgb
        
        return None

    def get_legal_actions(self) -> list:
        legal_moves = self.game_env.enumerate_legal()
        return [(r1, c1, r2, c2) for ((r1, c1, r2, c2), _) in legal_moves]

    def get_legal_actions_count(self) -> int:
        return len(self.game_env.enumerate_legal())