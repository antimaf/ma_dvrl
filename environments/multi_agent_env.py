import gym
import numpy as np
from gym import spaces
from typing import List, Dict, Tuple

class MultiAgentEnv(gym.Env):
    """Base class for multi-agent environments"""
    def __init__(self, num_agents: int, max_steps: int = 1000):
        super().__init__()
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define action and observation spaces for each agent
        self.action_spaces = {}
        self.observation_spaces = {}
        
    def reset(self) -> Dict[int, np.ndarray]:
        """Reset environment and return initial observations"""
        self.current_step = 0
        return {i: self._get_obs(i) for i in range(self.num_agents)}
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Execute one step for all agents"""
        self.current_step += 1
        
        # Execute actions and get next states
        next_obs = {i: self._get_obs(i) for i in range(self.num_agents)}
        rewards = {i: self._get_reward(i, actions) for i in range(self.num_agents)}
        dones = {i: self.current_step >= self.max_steps for i in range(self.num_agents)}
        infos = {i: {} for i in range(self.num_agents)}
        
        return next_obs, rewards, dones, infos
    
    def _get_obs(self, agent_id: int) -> np.ndarray:
        """Get observation for specific agent"""
        raise NotImplementedError
    
    def _get_reward(self, agent_id: int, actions: Dict[int, int]) -> float:
        """Get reward for specific agent"""
        raise NotImplementedError
