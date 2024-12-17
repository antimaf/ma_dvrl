import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from collections import deque

class PPONetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int):
        """Initialize PPO network with separate policy and value heads.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            num_actions: Number of possible actions
        """
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (action logits, value estimate)
        """
        shared_features = self.shared(x)
        return self.policy(shared_features), self.value(shared_features)

class PPOTrainer:
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_actions: int = 4,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 device: str = 'cuda'):
        """Initialize PPO trainer.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            num_actions: Number of possible actions
            lr: Learning rate
            gamma: Discount factor
            epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            device: Device to run on
        """
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.network = PPONetwork(input_dim, hidden_dim, num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []
        
    def select_action(self, state: torch.Tensor) -> Tuple[int, float, float]:
        """Select action using current policy.
        
        Args:
            state: Current state tensor
            
        Returns:
            Tuple of (selected action, action log probability, value estimate)
        """
        with torch.no_grad():
            logits, value = self.network(state)
            action_probs = F.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, value, log_prob, mask):
        """Store transition for later training.
        
        Args:
            state: State tensor
            action: Selected action
            reward: Received reward
            value: Value estimate
            log_prob: Log probability of action
            mask: Done mask (0 if terminal, 1 otherwise)
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.masks.append(mask)
    
    def train_batch(self) -> Dict[str, float]:
        """Train on collected transitions.
        
        Returns:
            Dictionary of training metrics
        """
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, device=self.device)
        old_log_probs = torch.tensor(self.log_probs, device=self.device)
        
        # Compute returns and advantages
        returns = self._compute_returns()
        returns = torch.tensor(returns, device=self.device)
        
        # Forward pass
        logits, values = self.network(states)
        values = values.squeeze()
        
        # Compute action probabilities
        action_probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        
        # Compute ratio and clipped ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
        
        # Compute losses
        advantages = returns - values.detach()
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -dist.entropy().mean()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
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
        
        for r, m in zip(reversed(self.rewards), reversed(self.masks)):
            running_return = r + self.gamma * running_return * m
            returns.insert(0, running_return)
            
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
