import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

class CardEmbedding(nn.Module):
    """Embeds poker cards into a learned representation."""
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(52, embedding_dim)  # 52 cards
        
    def forward(self, cards: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cards: Shape (batch_size, num_cards)
        Returns:
            embeddings: Shape (batch_size, num_cards, embedding_dim)
        """
        return self.embedding(cards)

class PokerBeliefEncoder(nn.Module):
    """Encodes poker game state into belief embeddings."""
    def __init__(
        self,
        card_dim: int = 32,
        hidden_dim: int = 128,
        belief_dim: int = 256,
        num_heads: int = 4
    ):
        super().__init__()
        self.card_embedding = CardEmbedding(card_dim)
        
        # Process cards using attention
        self.card_attention = nn.MultiheadAttention(
            embed_dim=card_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Process game state (chips, pot, stage)
        self.state_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # 2 chips + 1 pot + 1 stage
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Combine card and state information
        self.belief_encoder = nn.Sequential(
            nn.Linear(card_dim + hidden_dim, belief_dim),
            nn.ReLU(),
            nn.Linear(belief_dim, belief_dim)
        )
        
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs: Dictionary containing:
                - cards: Shape (batch_size, 7)
                - chips: Shape (batch_size, 2)
                - pot: Shape (batch_size, 1)
                - stage: Shape (batch_size,)
        Returns:
            belief_embedding: Shape (batch_size, belief_dim)
        """
        # Embed and attend to cards
        card_embeddings = self.card_embedding(obs['cards'])
        card_attn_out, _ = self.card_attention(
            card_embeddings,
            card_embeddings,
            card_embeddings
        )
        card_features = card_attn_out.mean(dim=1)  # Pool card features
        
        # Process game state
        state = torch.cat([
            obs['chips'],
            obs['pot'],
            obs['stage'].unsqueeze(-1).float()
        ], dim=-1)
        state_features = self.state_encoder(state)
        
        # Combine features
        combined_features = torch.cat([card_features, state_features], dim=-1)
        belief_embedding = self.belief_encoder(combined_features)
        
        return belief_embedding

class PokerOpponentModel(nn.Module):
    """Models opponent behavior in poker."""
    def __init__(self, belief_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # 4 actions
        )
    
    def forward(self, belief_embedding: torch.Tensor) -> torch.Tensor:
        """Predicts opponent action probabilities."""
        return F.softmax(self.network(belief_embedding), dim=-1)

class PokerDVRLPolicy(nn.Module):
    """Policy network for poker DVRL."""
    def __init__(self, belief_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # 4 actions
        )
        
        # Adaptive temperature for exploration
        self.log_temperature = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        belief_embedding: torch.Tensor,
        valid_actions: torch.Tensor,
        belief_entropy: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            belief_embedding: Shape (batch_size, belief_dim)
            valid_actions: Shape (batch_size, num_actions)
            belief_entropy: Shape (batch_size,)
        Returns:
            action_probs: Shape (batch_size, num_actions)
        """
        logits = self.network(belief_embedding)
        
        # Mask invalid actions
        logits = logits.masked_fill(~valid_actions.bool(), float('-inf'))
        
        # Apply temperature scaling
        temperature = torch.exp(self.log_temperature)
        if belief_entropy is not None:
            temperature = temperature * (1.0 + belief_entropy)
        
        return F.softmax(logits / temperature.unsqueeze(-1), dim=-1)

class PokerMADVRL(nn.Module):
    """Complete MA-DVRL model for poker."""
    def __init__(
        self,
        card_dim: int = 32,
        belief_dim: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 4
    ):
        super().__init__()
        
        # Create networks for each agent
        self.belief_encoders = nn.ModuleList([
            PokerBeliefEncoder(card_dim, hidden_dim, belief_dim, num_heads)
            for _ in range(2)  # Two players
        ])
        
        self.opponent_models = nn.ModuleList([
            PokerOpponentModel(belief_dim, hidden_dim)
            for _ in range(2)
        ])
        
        self.policies = nn.ModuleList([
            PokerDVRLPolicy(belief_dim, hidden_dim)
            for _ in range(2)
        ])
    
    def compute_belief_entropy(self, cards: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty in the belief state."""
        # Use number of unknown cards as a proxy for uncertainty
        unknown_cards = (cards == -1).float().sum(dim=-1)
        return unknown_cards / 7.0  # Normalize by total number of cards
    
    def forward(
        self,
        observations: Dict[int, Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Args:
            observations: Dict of observation tensors for each agent
        Returns:
            actions: Dict of action probabilities for each agent
            opponent_predictions: Dict of predicted opponent actions
        """
        actions = {}
        opponent_predictions = {}
        
        for i in range(2):
            # Convert observation tensors to proper type
            obs = {
                k: torch.as_tensor(v).float() if k != 'cards' else torch.as_tensor(v).long()
                for k, v in observations[i].items()
            }
            
            # Encode belief state
            belief_embedding = self.belief_encoders[i](obs)
            
            # Compute belief uncertainty
            belief_entropy = self.compute_belief_entropy(obs['cards'])
            
            # Predict opponent actions
            opponent_predictions[i] = self.opponent_models[i](belief_embedding)
            
            # Compute policy
            actions[i] = self.policies[i](
                belief_embedding,
                obs['valid_actions'],
                belief_entropy
            )
        
        return actions, opponent_predictions
    
    def get_loss(
        self,
        observations: Dict[int, Dict[str, torch.Tensor]],
        actions: Dict[int, torch.Tensor],
        rewards: Dict[int, torch.Tensor],
        next_observations: Dict[int, Dict[str, torch.Tensor]],
        dones: Dict[int, torch.Tensor],
        gamma: float = 0.99
    ) -> torch.Tensor:
        """Compute the loss for training."""
        total_loss = 0.0
        
        for i in range(2):
            # Current belief embedding
            obs = {
                k: torch.as_tensor(v).float() if k != 'cards' else torch.as_tensor(v).long()
                for k, v in observations[i].items()
            }
            belief_embedding = self.belief_encoders[i](obs)
            belief_entropy = self.compute_belief_entropy(obs['cards'])
            
            # Next belief embedding
            next_obs = {
                k: torch.as_tensor(v).float() if k != 'cards' else torch.as_tensor(v).long()
                for k, v in next_observations[i].items()
            }
            next_belief_embedding = self.belief_encoders[i](next_obs)
            
            # Policy loss
            policy_probs = self.policies[i](
                belief_embedding,
                obs['valid_actions'],
                belief_entropy
            )
            log_probs = torch.log(policy_probs + 1e-10)
            taken_log_probs = torch.sum(log_probs * actions[i], dim=-1)
            
            # Simple value estimation
            value = torch.sum(policy_probs * rewards[i], dim=-1)
            next_value = torch.sum(
                self.policies[i](
                    next_belief_embedding,
                    next_obs['valid_actions'],
                    belief_entropy
                ) * rewards[i],
                dim=-1
            )
            
            # Compute advantage
            advantage = rewards[i] + (1 - dones[i]) * gamma * next_value.detach() - value
            
            # Compute losses
            policy_loss = -(taken_log_probs * advantage.detach()).mean()
            value_loss = F.mse_loss(value, rewards[i] + gamma * next_value.detach())
            entropy_loss = -(policy_probs * log_probs).sum(dim=-1).mean()
            
            # Opponent modeling loss
            opponent_pred = self.opponent_models[i](belief_embedding)
            opponent_loss = F.cross_entropy(
                opponent_pred,
                actions[(i + 1) % 2].argmax(dim=-1)
            )
            
            # Combine losses
            agent_loss = (
                policy_loss +
                0.5 * value_loss +
                0.01 * entropy_loss +
                0.1 * opponent_loss
            )
            
            total_loss += agent_loss
        
        return total_loss / 2.0  # Average over agents
