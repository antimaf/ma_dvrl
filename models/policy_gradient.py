import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from collections import deque

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int):
        """Initialize policy network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            num_actions: Number of possible actions
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network.
        
        Args:
            x: Input tensor
            
        Returns:
            Action logits
        """
        return self.network(x)

class PolicyGradientTrainer:
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_actions: int = 4,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 entropy_coef: float = 0.01,
                 device: str = 'cuda'):
        """Initialize Policy Gradient trainer.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            num_actions: Number of possible actions
            lr: Learning rate
            gamma: Discount factor
            entropy_coef: Entropy bonus coefficient
            device: Device to run on
        """
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        self.network = PolicyNetwork(input_dim, hidden_dim, num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        self.saved_log_probs = []
        self.rewards = []
        
    def select_action(self, state: torch.Tensor) -> int:
        """Select action using current policy.
        
        Args:
            state: Current state tensor
            
        Returns:
            Selected action
        """
        with torch.no_grad():
            logits = self.network(state)
            action_probs = F.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            self.saved_log_probs.append(action_dist.log_prob(action))
            
        return action.item()
    
    def store_reward(self, reward: float):
        """Store reward for later training.
        
        Args:
            reward: Received reward
        """
        self.rewards.append(reward)
    
    def train_episode(self) -> Dict[str, float]:
        """Train on collected episode.
        
        Returns:
            Dictionary of training metrics
        """
        returns = self._compute_returns()
        returns = torch.tensor(returns, device=self.device)
        
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Add entropy bonus
        logits = self.network(torch.stack([s for s in self.saved_states]))
        action_probs = F.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        entropy_loss = -action_dist.entropy().mean()
        
        # Total loss
        loss = policy_loss + self.entropy_coef * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear memory
        self.saved_log_probs = []
        self.rewards = []
        self.saved_states = []
        
        return {
            'policy_loss': policy_loss.item(),
            'entropy': entropy_loss.item(),
            'total_loss': loss.item()
        }
    
    def _compute_returns(self) -> np.ndarray:
        """Compute discounted returns.
        
        Returns:
            Array of discounted returns
        """
        returns = []
        running_return = 0
        
        for r in reversed(self.rewards):
            running_return = r + self.gamma * running_return
            returns.insert(0, running_return)
            
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
