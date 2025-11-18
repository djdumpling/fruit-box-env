"""
Q-learning training script with augment factor support (gamma > 1).

Template for future implementation. This script provides a structure for Q-learning
with support for augment factor (gamma > 1) to favor later rewards.

Usage:
    python rl/train_q_learning.py --checkpoint checkpoints/policy_sft_epoch30.pt --gamma 1.005
"""

import sys
from pathlib import Path
# add project root to path for imports (go up 2 levels from rl/train/train_q_learning.py)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
import wandb

from rl.envs.sum10_env import Sum10GymEnv
from rl.envs.split_wrapper import TwoPhaseWrapper
from rl.models.policy import CNNPolicy
from fruit_box import Sum10Env


@dataclass
class Config:
    """Q-learning training config"""
    # data collection
    num_envs: int = 16
    batch_size: int = 64
    replay_buffer_size: int = 100000
    
    # Q-learning hyperparameters
    gamma: float = 1.005  # augment factor (gamma > 1) to favor later rewards
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_interval: int = 100
    learning_rate: float = 1e-4
    
    # training
    max_updates: int = 5000
    max_steps_per_episode: int = 85
    
    # other
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 500
    load_checkpoint: Optional[str] = None


class QNetwork(nn.Module):
    """Q-network architecture (similar to policy but outputs Q-values for all actions)"""
    
    def __init__(self, obs_shape: Tuple[int, int, int], action_dim: int):
        super().__init__()
        # reuse CNNPolicy architecture but modify output head
        # TODO: implement Q-network based on CNNPolicy structure
        # output should be [batch_size, action_dim] Q-values
        pass
    
    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values for all actions"""
        # TODO: implement forward pass
        # return Q(s, a) for all actions a
        pass


class ReplayBuffer:
    """Experience replay buffer for Q-learning"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, mask):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done, mask))
    
    def sample(self, batch_size: int):
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        # TODO: convert to tensors
        return batch
    
    def __len__(self):
        return len(self.buffer)


def epsilon_greedy_action(q_network: QNetwork, obs: torch.Tensor, action_mask: torch.Tensor, epsilon: float, device: torch.device) -> int:
    """Select action using epsilon-greedy policy"""
    if random.random() < epsilon:
        # random action from valid actions
        valid_actions = torch.nonzero(action_mask, as_tuple=False).squeeze(-1)
        if valid_actions.numel() == 0:
            return 0
        return valid_actions[random.randint(0, len(valid_actions) - 1)].item()
    else:
        # greedy action
        with torch.no_grad():
            q_values = q_network(obs.unsqueeze(0).to(device), action_mask.unsqueeze(0).to(device))
            # mask invalid actions
            q_values[0][~action_mask] = float('-inf')
            return q_values.argmax().item()


def compute_q_loss(q_network: QNetwork, target_network: QNetwork, batch, gamma: float, device: torch.device) -> Tuple[torch.Tensor, Dict]:
    """Compute Q-learning loss with augment factor support"""
    # TODO: implement Q-learning update
    # Q(s, a) = r + gamma * max_a' Q_target(s', a')
    # where gamma can be > 1 (augment factor)
    
    # Note: gamma > 1 amplifies future rewards, encouraging the agent to
    # delay good moves until later (minimal area strategy)
    
    states, actions, rewards, next_states, dones, masks = batch
    
    # current Q-values
    q_values = q_network(states, masks)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # target Q-values
    with torch.no_grad():
        next_q_values = target_network(next_states, masks)
        next_q_values[~masks] = float('-inf')
        next_q_value = next_q_values.max(1)[0]
        target_q_value = rewards + (gamma ** (1 - dones.float())) * next_q_value
    
    # loss
    loss = nn.MSELoss()(q_value, target_q_value)
    
    info = {
        "q_loss": loss.item(),
        "mean_q_value": q_value.mean().item(),
        "mean_target_q_value": target_q_value.mean().item(),
    }
    
    return loss, info


def train(config: Config, use_wandb: bool = True):
    """Main training loop for Q-learning"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {config.seed} | Gamma: {config.gamma}")
    
    # set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # create Q-network and target network
    q_network = QNetwork(obs_shape=(4, 10, 17), action_dim=170).to(device)
    target_network = QNetwork(obs_shape=(4, 10, 17), action_dim=170).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    
    # load checkpoint if provided
    if config.load_checkpoint:
        print(f"Loading checkpoint from {config.load_checkpoint}...")
        checkpoint = torch.load(config.load_checkpoint, map_location=device)
        # TODO: adapt SFT checkpoint to Q-network if needed
        q_network.load_state_dict(checkpoint)
        target_network.load_state_dict(q_network.state_dict())
        print("Checkpoint loaded successfully!")
    
    # create optimizer
    optimizer = torch.optim.Adam(q_network.parameters(), lr=config.learning_rate)
    
    # create replay buffer
    replay_buffer = ReplayBuffer(config.replay_buffer_size)
    
    # create environments
    envs = []
    for i in range(config.num_envs):
        env = Sum10GymEnv(seed=config.seed + i)
        env = TwoPhaseWrapper(env, curriculum_legal_only=False, curriculum_updates=0)
        envs.append(env)
    
    # initialize wandb
    if use_wandb:
        wandb.init(
            project="fruit-box-rl",
            name=f"qlearning_gamma{config.gamma}",
            config={
                "gamma": config.gamma,
                "epsilon_start": config.epsilon_start,
                "epsilon_end": config.epsilon_end,
                "epsilon_decay": config.epsilon_decay,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "replay_buffer_size": config.replay_buffer_size,
            }
        )
    
    # training loop
    epsilon = config.epsilon_start
    global_step = 0
    
    for update in range(config.max_updates):
        # TODO: implement training loop
        # 1. Collect experiences from environments
        # 2. Store in replay buffer
        # 3. Sample batch and update Q-network
        # 4. Update target network periodically
        # 5. Decay epsilon
        # 6. Log metrics
        
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        
        if use_wandb:
            wandb.log({
                "update": update,
                "epsilon": epsilon,
            }, step=update)
    
    print("Training complete!")
    
    # save final checkpoint
    final_checkpoint_path = f"{config.checkpoint_dir}/qnetwork_final.pt"
    torch.save(q_network.state_dict(), final_checkpoint_path)
    print(f"Saved final checkpoint: {final_checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Q-learning training with augment factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gamma", type=float, default=1.005, help="Discount/augment factor (gamma > 1 for augment)")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Starting epsilon for epsilon-greedy")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="Ending epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-updates", type=int, default=5000, help="Maximum training updates")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to checkpoint to load")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    config = Config(
        seed=args.seed,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_updates=args.max_updates,
        load_checkpoint=args.load_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    train(config, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()

