import numpy as np
from gym import spaces
from .multi_agent_env import MultiAgentEnv
from typing import Dict, List, Tuple, Optional

class PokerEnv(MultiAgentEnv):
    """Two-player Texas Hold'em Poker Environment"""
    
    ACTIONS = {'FOLD': 0, 'CALL': 1, 'RAISE': 2, 'CHECK': 3}
    STAGES = ['PREFLOP', 'FLOP', 'TURN', 'RIVER', 'SHOWDOWN']
    
    def __init__(self, initial_chips: int = 1000):
        super().__init__(num_agents=2)  # Two-player heads-up poker
        self.initial_chips = initial_chips
        
        # Override action and observation spaces
        for i in range(self.num_agents):
            self.action_spaces[i] = spaces.Discrete(len(self.ACTIONS))
            # Observation space: [player_cards (2), community_cards (5), chips, pot, stage]
            self.observation_spaces[i] = spaces.Dict({
                'cards': spaces.Box(low=0, high=51, shape=(7,), dtype=np.int32),
                'chips': spaces.Box(low=0, high=float('inf'), shape=(2,), dtype=np.float32),
                'pot': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
                'stage': spaces.Discrete(len(self.STAGES)),
                'valid_actions': spaces.MultiBinary(len(self.ACTIONS))
            })
        
        self.reset()
    
    def reset(self) -> Dict[int, Dict]:
        """Reset the environment for a new hand."""
        self.deck = list(range(52))
        np.random.shuffle(self.deck)
        
        # Deal cards
        self.hands = {
            i: self.deck[i*2:(i+1)*2] for i in range(self.num_agents)
        }
        self.community_cards = []
        
        # Reset game state
        self.chips = {i: self.initial_chips for i in range(self.num_agents)}
        self.pot = 0
        self.current_player = 0
        self.stage_idx = 0
        self.current_bet = 0
        self.last_raise = 0
        
        return {i: self._get_obs(i) for i in range(self.num_agents)}
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Execute one step of the environment."""
        action = actions[self.current_player]
        rewards = {i: 0.0 for i in range(self.num_agents)}
        done = False
        
        # Process action
        if action == self.ACTIONS['FOLD']:
            # Current player folds, opponent wins pot
            opponent = (self.current_player + 1) % 2
            rewards[opponent] = self.pot
            rewards[self.current_player] = -self.pot
            done = True
        
        elif action == self.ACTIONS['CALL']:
            # Match the current bet
            call_amount = min(self.current_bet - self.last_raise, 
                            self.chips[self.current_player])
            self.chips[self.current_player] -= call_amount
            self.pot += call_amount
            self.last_raise = self.current_bet
        
        elif action == self.ACTIONS['RAISE']:
            # Raise the current bet
            raise_amount = min(self.current_bet * 2, self.chips[self.current_player])
            self.chips[self.current_player] -= raise_amount
            self.pot += raise_amount
            self.current_bet = raise_amount
        
        # Move to next player or stage
        if not done:
            self.current_player = (self.current_player + 1) % 2
            if self.current_player == 0:  # Round complete
                if self.stage_idx < len(self.STAGES) - 1:
                    self._deal_community_cards()
                    self.stage_idx += 1
                else:
                    # Showdown
                    winner = self._determine_winner()
                    rewards[winner] = self.pot
                    rewards[(winner + 1) % 2] = -self.pot
                    done = True
        
        observations = {i: self._get_obs(i) for i in range(self.num_agents)}
        dones = {i: done for i in range(self.num_agents)}
        infos = {i: {} for i in range(self.num_agents)}
        
        return observations, rewards, dones, infos
    
    def _get_obs(self, agent_id: int) -> Dict:
        """Get observation for specific agent."""
        # Combine player's cards with visible community cards
        cards = np.full(7, -1)  # 2 player cards + 5 community cards
        cards[:2] = self.hands[agent_id]
        cards[2:2+len(self.community_cards)] = self.community_cards
        
        # Determine valid actions
        valid_actions = np.zeros(len(self.ACTIONS))
        valid_actions[self.ACTIONS['FOLD']] = 1
        valid_actions[self.ACTIONS['CALL']] = 1 if self.current_bet > self.last_raise else 0
        valid_actions[self.ACTIONS['RAISE']] = 1 if self.chips[agent_id] > 0 else 0
        valid_actions[self.ACTIONS['CHECK']] = 1 if self.current_bet == self.last_raise else 0
        
        return {
            'cards': cards,
            'chips': np.array([self.chips[i] for i in range(self.num_agents)]),
            'pot': np.array([self.pot]),
            'stage': self.stage_idx,
            'valid_actions': valid_actions
        }
    
    def _deal_community_cards(self):
        """Deal community cards based on current stage."""
        if self.stage_idx == 0:  # Flop
            self.community_cards = self.deck[4:7]
        elif self.stage_idx == 1:  # Turn
            self.community_cards.append(self.deck[7])
        elif self.stage_idx == 2:  # River
            self.community_cards.append(self.deck[8])
    
    def _determine_winner(self) -> int:
        """Determine the winner of the hand."""
        # Simplified poker hand evaluation
        # In a real implementation, this would evaluate poker hands
        # For now, just compare the highest card
        hand0 = max(self.hands[0] + self.community_cards)
        hand1 = max(self.hands[1] + self.community_cards)
        return 0 if hand0 > hand1 else 1
