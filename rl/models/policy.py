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
        
        # sum prediction head: predicts rectangle sum for each extent action
        # Only used in Phase-1 (extent selection)
        self.sum_prediction_head = nn.Linear(256, action_dim)  # sum prediction for each extent
        
        # Phase-1 head: concatenates features with sum predictions
        # Input: 256 (features) + action_dim (sum predictions) = 256 + action_dim
        self.phase1_head = nn.Linear(256 + action_dim, action_dim)  # extent selection
        
        # value head
        self.value_head = nn.Linear(256, 1)
    
    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            obs: [batch_size, 4, 10, 17] observation tensor
                Channel 3 (phase_mask) indicates phase: 0.0 = Phase-0, 1.0 = Phase-1
            action_mask: [batch_size, action_dim] binary mask (1 for valid actions)
        
        Returns:
            logits: [batch_size, action_dim] policy logits
            value: [batch_size, 1] value estimate
            sum_predictions: [batch_size, action_dim] predicted rectangle sums (for Phase-1, zeros for Phase-0)
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
        
        # Phase-0: anchor selection (no sum prediction needed)
        phase0_logits = self.phase0_head(x)  # [batch, action_dim]
        
        # Phase-1: extent selection (with sum prediction)
        # First, predict sums for all extent candidates
        sum_predictions = self.sum_prediction_head(x)  # [batch, action_dim]
        
        # Concatenate features with sum predictions for Phase-1 head
        phase1_features = torch.cat([x, sum_predictions], dim=1)  # [batch, 256 + action_dim]
        phase1_logits = self.phase1_head(phase1_features)  # [batch, action_dim]
        
        # Select logits based on phase
        # use torch.where to select: if is_phase1, use phase1_logits, else phase0_logits
        logits = torch.where(
            is_phase1.unsqueeze(-1),  # [batch, 1] - broadcast to action_dim
            phase1_logits,
            phase0_logits
        )  # [batch, action_dim]
        # Clamp logits to avoid saturation that leads to NaNs in downstream losses
        logits = torch.clamp(logits, min=-8.0, max=8.0)
        
        # For Phase-0, set sum predictions to zero (not used)
        sum_predictions = torch.where(
            is_phase1.unsqueeze(-1),  # [batch, 1] - broadcast to action_dim
            sum_predictions,
            torch.zeros_like(sum_predictions)
        )  # [batch, action_dim]
        
        # apply action mask
        if action_mask is not None:
            # set invalid actions to very negative value
            logits = logits.masked_fill(~action_mask, -1e9)
            # also mask sum predictions for invalid actions
            sum_predictions = sum_predictions.masked_fill(~action_mask, 0.0)
        
        # value estimate
        value = self.value_head(x)  # [batch, 1]
        
        return logits, value, sum_predictions
    
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
            action: [batch_size] sampled action indices (in original space, not compact)
            logprob: [batch_size] log probabilities of sampled actions
            value: [batch_size, 1] value estimates
        """
        logits, value, sum_predictions = self.forward(obs, action_mask)
        
        # Sample action from valid actions only (matching test_sft.py behavior)
        # This ensures we never sample invalid actions, even with numerical precision issues
        batch_size = obs.size(0)
        actions = []
        logprobs = []
        
        for b in range(batch_size):
            if action_mask is not None:
                mask = action_mask[b]  # [action_dim]
                valid_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                if valid_indices.numel() == 0:
                    # fallback: use argmax on full logits (shouldn't happen)
                    action = logits[b].argmax().item()
                    dist = torch.distributions.Categorical(logits=logits[b])
                    logprob = dist.log_prob(torch.tensor(action, device=obs.device))
                else:
                    # ensure valid_indices is 1D
                    if valid_indices.dim() == 0:
                        valid_indices = valid_indices.unsqueeze(0)
                    # extract logits at valid positions only
                    valid_logits = logits[b][valid_indices]
                    # sample from valid actions
                    dist = torch.distributions.Categorical(logits=valid_logits)
                    action_compact = dist.sample()
                    # map back to original index space
                    action = valid_indices[action_compact].item()
                    # compute logprob using original logits (needed for PPO)
                    dist_full = torch.distributions.Categorical(logits=logits[b])
                    logprob = dist_full.log_prob(torch.tensor(action, device=obs.device))
            else:
                # no mask: sample from all actions
                dist = torch.distributions.Categorical(logits=logits[b])
                action = dist.sample().item()
                logprob = dist.log_prob(torch.tensor(action, device=obs.device))
            
            actions.append(action)
            logprobs.append(logprob)
        
        actions_tensor = torch.tensor(actions, device=obs.device, dtype=torch.long)
        logprobs_tensor = torch.stack(logprobs)
        
        return actions_tensor, logprobs_tensor, value

