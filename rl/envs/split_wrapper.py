"""Two-phase wrapper for action factorization: Phase-0 selects anchor, Phase-1 selects extent."""
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
import gymnasium as gym
from gymnasium import spaces

from fruit_box import Sum10Env


class TwoPhaseWrapper(gym.Wrapper):
    """Wrapper that splits action selection into two phases.
    
    Phase 0: Select anchor (r1, c1) from 170 cells
    Phase 1: Select extent (r2, c2) with constraints r2>=r1, c2>=c1
    """
    
    def __init__(
        self,
        env,
        curriculum_legal_only: bool = True,
        curriculum_updates: int = 200,
    ):
        super().__init__(env)
        self.curriculum_legal_only = curriculum_legal_only
        self.curriculum_updates = curriculum_updates
        self.current_update = 0
        
        # Phase state: 0 = selecting anchor, 1 = selecting extent
        self.phase = 0
        self.selected_anchor = None  # (r1, c1) tuple
        
        # Access underlying Sum10Env
        if hasattr(env, 'game_env'):
            self.game_env = env.game_env
        elif isinstance(env, Sum10Env):
            self.game_env = env
        else:
            raise ValueError("env must have game_env attribute or be Sum10Env")
        
        # Phase-0: select anchor from 170 cells (10*17)
        self.phase0_action_dim = 170
        
        # Phase-1: variable action space (depends on anchor)
        # Max is when anchor is (0,0): 10*17 = 170
        self.phase1_max_action_dim = 170
        
        # Observation: [4, 10, 17] tensor
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4, 10, 17), dtype=np.float32
        )
        
        # Action space: Discrete(170) for Phase-0, variable for Phase-1
        self.action_space = spaces.Discrete(self.phase0_action_dim)
    
    def anchor_to_flat_idx(self, r1: int, c1: int) -> int:
        """Convert anchor (r1, c1) to flat index [0, 169]."""
        return r1 * 17 + c1
    
    def flat_idx_to_anchor(self, idx: int) -> Tuple[int, int]:
        """Convert flat index [0, 169] to anchor (r1, c1)."""
        r1 = idx // 17
        c1 = idx % 17
        return (r1, c1)
    
    def extent_to_flat_idx(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """Convert extent (r2, c2) to flat index given anchor (r1, c1).
        
        Valid extents: r2 in [r1, 9], c2 in [c1, 16]
        Flat index: (r2 - r1) * (17 - c1) + (c2 - c1)
        """
        if not (r1 <= r2 < 10 and c1 <= c2 < 17):
            raise ValueError(f"Invalid extent: anchor=({r1},{c1}), extent=({r2},{c2})")
        dr = r2 - r1
        dc = c2 - c1
        width = 17 - c1
        return dr * width + dc
    
    def flat_idx_to_extent(self, r1: int, c1: int, idx: int) -> Tuple[int, int]:
        """Convert flat index to extent (r2, c2) given anchor (r1, c1)."""
        width = 17 - c1
        dr = idx // width
        dc = idx % width
        r2 = r1 + dr
        c2 = c1 + dc
        return (r2, c2)
    
    def get_action_mask(self) -> torch.Tensor:
        """Get action mask for current phase.
        
        Returns:
            mask: [action_dim] binary tensor, 1 for valid actions
        """
        if self.phase == 0:
            # Phase-0: all anchors are valid
            return torch.ones(self.phase0_action_dim, dtype=torch.bool)
        else:
            # Phase-1: mask based on r2>=r1, c2>=c1
            r1, c1 = self.selected_anchor
            action_dim = (10 - r1) * (17 - c1)
            mask = torch.ones(action_dim, dtype=torch.bool)
            
            # Gradual curriculum annealing instead of hard switch
            if self.curriculum_legal_only and self.current_update < self.curriculum_updates * 2:
                legal_mask = self.get_legal_only_mask()
                
                if self.current_update < self.curriculum_updates:
                    # Strict phase: only legal actions
                    mask = mask & legal_mask
                else:
                    # Annealing phase: gradually mix in illegal actions
                    anneal_progress = (self.current_update - self.curriculum_updates) / self.curriculum_updates
                    # Use legal actions + sample illegal actions with increasing probability
                    illegal_mask = ~legal_mask
                    # Start allowing 10% of illegal actions, gradually increase to 100%
                    num_illegal = illegal_mask.sum().item()
                    if num_illegal > 0:
                        num_to_allow = max(1, int(num_illegal * 0.1 * anneal_progress))
                        illegal_indices = torch.nonzero(illegal_mask, as_tuple=False).squeeze(-1)
                        if len(illegal_indices) > 0:
                            # Randomly sample illegal actions to allow
                            num_to_allow = min(num_to_allow, len(illegal_indices))
                            perm = torch.randperm(len(illegal_indices))[:num_to_allow]
                            selected_illegal = illegal_indices[perm]
                            mask[selected_illegal] = True
            
            return mask
    
    def get_legal_only_mask(self) -> torch.Tensor:
        """Get legal-only mask for Phase-1 (curriculum).
        
        Only includes rectangles where sum == 10.
        """
        if self.phase == 0:
            return torch.ones(self.phase0_action_dim, dtype=torch.bool)
        
        r1, c1 = self.selected_anchor
        action_dim = (10 - r1) * (17 - c1)
        mask = torch.zeros(action_dim, dtype=torch.bool)
        
        # Check each valid extent
        for idx in range(action_dim):
            r2, c2 = self.flat_idx_to_extent(r1, c1, idx)
            if self.game_env.box_sum(r1, c1, r2, c2) == 10:
                mask[idx] = True
        
        return mask
    
    def _build_observation(self) -> np.ndarray:
        """Build 4-channel observation tensor."""
        grid = self.game_env.grid.astype(np.float32)
        
        # Channel 0: normalized values
        value_norm = grid / 9.0
        
        # Channel 1: nonzero mask
        nonzero_mask = (grid > 0).astype(np.float32)
        
        # Channel 2: anchor mask (zeros in Phase-0, selected anchor=1 in Phase-1)
        anchor_mask = np.zeros((10, 17), dtype=np.float32)
        if self.phase == 1 and self.selected_anchor is not None:
            r1, c1 = self.selected_anchor
            anchor_mask[r1, c1] = 1.0
        
        # Channel 3: phase mask (all zeros in Phase-0, all ones in Phase-1)
        phase_mask = np.full((10, 17), float(self.phase), dtype=np.float32)
        
        obs = np.stack([value_norm, nonzero_mask, anchor_mask, phase_mask], axis=0)
        return obs
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset environment and return Phase-0 observation."""
        obs, info = self.env.reset(seed=seed, options=options)
        
        self.phase = 0
        self.selected_anchor = None
        
        obs_4ch = self._build_observation()
        info["phase"] = 0
        
        return torch.from_numpy(obs_4ch), info
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        """Step environment with phase-aware action.
        
        Phase-0: action is anchor index, transition to Phase-1
        Phase-1: action is extent index, execute move and transition to Phase-0
        """
        if self.phase == 0:
            # Phase-0: select anchor
            r1, c1 = self.flat_idx_to_anchor(action)
            self.selected_anchor = (r1, c1)
            self.phase = 1
            
            # Return observation, no reward yet
            obs_4ch = self._build_observation()
            info = {"phase": 1, "anchor": (r1, c1)}
            
            return torch.from_numpy(obs_4ch), 0.0, False, False, info
        
        else:
            # Phase-1: select extent and execute move
            r1, c1 = self.selected_anchor
            r2, c2 = self.flat_idx_to_extent(r1, c1, action)
            
            # Execute move on underlying environment
            # Convert to MultiDiscrete action format
            multi_action = np.array([r1, c1, r2, c2], dtype=np.int32)
            obs, reward, terminated, truncated, info = self.env.step(multi_action)
            
            # Apply curriculum penalty if needed
            if self.current_update >= self.curriculum_updates:
                # Check if move was illegal (sum != 10)
                if not info.get("valid", True):
                    reward += -0.05  # penalty for illegal rectangle
            
            # Reset phase state
            self.phase = 0
            self.selected_anchor = None
            
            # If episode ended, reset phase state
            if terminated or truncated:
                self.phase = 0
                self.selected_anchor = None
            
            obs_4ch = self._build_observation()
            info["phase"] = 0
            
            return torch.from_numpy(obs_4ch), reward, terminated, truncated, info
    
    def set_curriculum_update(self, update: int):
        """Update curriculum progress."""
        self.current_update = update

