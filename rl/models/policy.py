"""CNN policy network for Fruit Box environment."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CNNPolicy(nn.Module):
    """CNN policy network with policy and value heads.
    
    Architecture:
    - Conv2d(4, 32, 3x3) → GroupNorm → GELU → Conv2d(32, 64, 3x3) → GroupNorm → GELU
    - Flatten → Linear(64*8*15, 256) → GELU → LayerNorm
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
        
        # convolutional layers
        # input: (4, 10, 17)
        # conv1: 3x3, no padding → (32, 8, 15)
        self.conv1 = nn.Conv2d(obs_shape[0], 32, kernel_size=3, padding=0)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=32)
        # conv2: 3x3, padding=1 → (64, 8, 15) (maintains spatial size)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=64)
        
        # flattened size: 64 * 8 * 15 = 7680
        self.flattened_size = 64 * 8 * 15
        
        # feature extractor
        self.feature_extractor = nn.Sequential(
            self.conv1,
            self.gn1,
            nn.GELU(),
            self.conv2,
            self.gn2,
            nn.GELU(),
        )
        
        # fully connected layers
        self.fc = nn.Linear(self.flattened_size, 256)
        self.ln = nn.LayerNorm(256)
        
        # separate policy heads for Phase-0 (anchor selection) and Phase-1 (extent selection)
        self.phase0_head = nn.Linear(256, action_dim)  # anchor selection
        self.phase1_head = nn.Linear(256, action_dim)  # extent selection
        
        # value head
        self.value_head = nn.Linear(256, 1)
    
    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            obs: [batch_size, 4, 10, 17] observation tensor
                Channel 3 (phase_mask) indicates phase: 0.0 = Phase-0, 1.0 = Phase-1
            action_mask: [batch_size, action_dim] binary mask (1 for valid actions)
        
        Returns:
            logits: [batch_size, action_dim] policy logits
            value: [batch_size, 1] value estimate
        """
        # extract features
        x = self.feature_extractor(obs)  # [batch, 64, 8, 15]
        x = x.view(x.size(0), -1)  # [batch, 7680]
        x = F.gelu(self.fc(x))  # [batch, 256]
        x = self.ln(x)  # [batch, 256] - LayerNorm before heads
        
        # determine phase from observation (channel 3, any pixel will do)
        # phase_mask is 0.0 for Phase-0, 1.0 for Phase-1
        phase_indicator = obs[:, 3, 0, 0]  # [batch] - take any pixel from phase mask channel
        is_phase1 = phase_indicator > 0.5  # [batch] - True for Phase-1
        
        # use separate heads based on phase
        phase0_logits = self.phase0_head(x)  # [batch, action_dim]
        phase1_logits = self.phase1_head(x)  # [batch, action_dim]
        
        # select logits based on phase
        # use torch.where to select: if is_phase1, use phase1_logits, else phase0_logits
        logits = torch.where(
            is_phase1.unsqueeze(-1),  # [batch, 1] - broadcast to action_dim
            phase1_logits,
            phase0_logits
        )  # [batch, action_dim]
        
        # apply action mask
        if action_mask is not None:
            # set invalid actions to very negative value
            logits = logits.masked_fill(~action_mask, -1e9)
        
        # value estimate
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

