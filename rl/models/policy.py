"""CNN policy network for Fruit Box environment."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CNNPolicy(nn.Module):
    """CNN policy network with policy and value heads.
    
    Architecture:
    - Conv2d(4, 32, 3x3) → ReLU → Conv2d(32, 64, 3x3) → ReLU
    - Flatten → Linear(64*8*15, 256) → ReLU
    - Policy head: Linear(256, action_dim)
    - Value head: Linear(256, 1)
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, int, int] = (4, 10, 17),
        action_dim: int = 170,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        
        # Convolutional layers
        # Input: (4, 10, 17)
        # Conv1: 3x3, no padding → (32, 8, 15)
        self.conv1 = nn.Conv2d(obs_shape[0], 32, kernel_size=3, padding=0)
        # Conv2: 3x3, padding=1 → (64, 8, 15) (maintains spatial size)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Flattened size: 64 * 8 * 15 = 7680
        self.flattened_size = 64 * 8 * 15
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
        )
        
        # Fully connected layers
        self.fc = nn.Linear(self.flattened_size, 256)
        
        # Policy head
        self.policy_head = nn.Linear(256, action_dim)
        
        # Value head
        self.value_head = nn.Linear(256, 1)
    
    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            obs: [batch_size, 4, 10, 17] observation tensor
            action_mask: [batch_size, action_dim] binary mask (1 for valid actions)
        
        Returns:
            logits: [batch_size, action_dim] policy logits
            value: [batch_size, 1] value estimate
        """
        # Extract features
        x = self.feature_extractor(obs)  # [batch, 64, 8, 15]
        x = x.view(x.size(0), -1)  # [batch, 7680]
        x = F.relu(self.fc(x))  # [batch, 256]
        
        # Policy logits
        logits = self.policy_head(x)  # [batch, action_dim]
        
        # Apply action mask
        if action_mask is not None:
            # Set invalid actions to very negative value
            logits = logits.masked_fill(~action_mask, -1e9)
        
        # Value estimate
        value = self.value_head(x)  # [batch, 1]
        
        return logits, value
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, logprob, and value.
        
        Args:
            obs: [batch_size, 4, 10, 17] observation tensor
            action_mask: [batch_size, action_dim] binary mask
        
        Returns:
            action: [batch_size] sampled action indices
            logprob: [batch_size] log probabilities of sampled actions
            value: [batch_size, 1] value estimates
        """
        logits, value = self.forward(obs, action_mask)
        
        # Sample action
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        
        return action, logprob, value

