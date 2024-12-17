import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class CFRPlusTrainer:
    def __init__(self, num_actions: int = 4):
        """Initialize CFR+ trainer for poker.
        
        Args:
            num_actions: Number of possible actions
        """
        self.num_actions = num_actions
        self.regret_sum = defaultdict(lambda: np.zeros(num_actions))
        self.strategy_sum = defaultdict(lambda: np.zeros(num_actions))
        self.iteration = 0
        
    def _get_strategy(self, info_set: str) -> np.ndarray:
        """Get current strategy for an information set.
        
        Args:
            info_set: String representation of information set
            
        Returns:
            Probability distribution over actions
        """
        regrets = self.regret_sum[info_set]
        positive_regrets = np.maximum(regrets, 0)
        sum_positive_regret = np.sum(positive_regrets)
        
        if sum_positive_regret > 0:
            return positive_regrets / sum_positive_regret
        else:
            return np.ones(self.num_actions) / self.num_actions
            
    def get_action(self, info_set: str) -> int:
        """Get action according to current strategy.
        
        Args:
            info_set: String representation of information set
            
        Returns:
            Selected action index
        """
        strategy = self._get_strategy(info_set)
        return np.random.choice(len(strategy), p=strategy)
        
    def update(self, info_set: str, action: int, regret: float):
        """Update regret and strategy sums.
        
        Args:
            info_set: String representation of information set
            action: Taken action
            regret: Computed regret for the action
        """
        self.regret_sum[info_set][action] += regret
        strategy = self._get_strategy(info_set)
        self.strategy_sum[info_set] += strategy * (self.iteration + 1)
        self.iteration += 1
        
    def get_average_strategy(self, info_set: str) -> np.ndarray:
        """Get average strategy across all iterations.
        
        Args:
            info_set: String representation of information set
            
        Returns:
            Average strategy as probability distribution
        """
        strategy_sum = self.strategy_sum[info_set]
        total = np.sum(strategy_sum)
        
        if total > 0:
            return strategy_sum / total
        else:
            return np.ones(self.num_actions) / self.num_actions
            
    def train_iteration(self, env) -> Tuple[float, Dict[str, float]]:
        """Run one iteration of CFR+.
        
        Args:
            env: Poker environment
            
        Returns:
            Tuple of (utility, metrics dictionary)
        """
        obs = env.reset()
        done = False
        total_reward = 0
        metrics = defaultdict(float)
        
        while not done:
            # Get current player's info set
            current_player = env.current_player
            info_set = self._get_info_set_string(obs[current_player])
            
            # Get action from current strategy
            action = self.get_action(info_set)
            
            # Take action
            next_obs, rewards, dones, _ = env.step({current_player: action})
            
            # Update metrics
            total_reward += rewards[0]  # Track rewards for player 0
            metrics['regret_sum'] = float(np.mean([np.sum(np.abs(r)) for r in self.regret_sum.values()]))
            
            # Compute and update regrets
            if not done:
                counterfactual_values = self._compute_counterfactual_values(env, next_obs)
                regret = counterfactual_values[action] - np.mean(counterfactual_values)
                self.update(info_set, action, regret)
            
            obs = next_obs
            done = any(dones.values())
            
        return total_reward, metrics
    
    def _get_info_set_string(self, obs: Dict) -> str:
        """Convert observation to information set string.
        
        Args:
            obs: Observation dictionary
            
        Returns:
            String representation of information set
        """
        # Combine relevant observation components into a string
        components = []
        if 'hand' in obs:
            components.append(f"hand:{obs['hand']}")
        if 'board' in obs:
            components.append(f"board:{obs['board']}")
        if 'pot' in obs:
            components.append(f"pot:{obs['pot']}")
        if 'valid_actions' in obs:
            components.append(f"valid:{obs['valid_actions']}")
        
        return "|".join(components)
    
    def _compute_counterfactual_values(self, env, obs: Dict) -> np.ndarray:
        """Compute counterfactual values for all actions.
        
        Args:
            env: Poker environment
            obs: Current observation
            
        Returns:
            Array of values for each action
        """
        values = np.zeros(self.num_actions)
        current_player = env.current_player
        
        # For each action, simulate outcome
        for action in range(self.num_actions):
            if obs[current_player]['valid_actions'][action]:
                env_copy = env.clone()  # Assuming environment supports cloning
                next_obs, rewards, dones, _ = env_copy.step({current_player: action})
                values[action] = rewards[current_player]
                
        return values
