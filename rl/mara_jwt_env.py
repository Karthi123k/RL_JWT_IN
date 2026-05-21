# mara_jwt_env.py
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv
from pqc_env import PQCEnv


class MARAJWTEnv(ParallelEnv):
    """
    MARA-JWT: Multi-Agent Reinforcement Learning for JWT PQC Algorithm Selection
    Each agent has specialized observation and reward
    """
    
    metadata = {"render_modes": ["human"], "name": "mara_jwt_v0"}
    
    def __init__(self):
        super().__init__()
        
        # Base PQC environment
        self.base_env = PQCEnv()
        
        # Define specialized agents
        self.agents = [
            "security_agent",
            "performance_agent", 
            "resource_agent",
            "traffic_agent",
            "coordinator"
        ]
        
        self.possible_agents = self.agents.copy()
        
        # Different observation spaces for each agent
        self.observation_spaces = {
            "security_agent": Box(low=0, high=1, shape=(3,), dtype=np.float32),
            "performance_agent": Box(low=0, high=1, shape=(3,), dtype=np.float32),
            "resource_agent": Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "traffic_agent": Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "coordinator": Box(low=0, high=1, shape=(10,), dtype=np.float32),
        }
        
        # Same action space for all agents
        self.action_spaces = {
            agent: Discrete(self.base_env.action_space.n) 
            for agent in self.agents
        }
        
        # Coordinator weights
        self.coordinator_weights = {
            "security_agent": 0.35,
            "performance_agent": 0.35,
            "resource_agent": 0.15,
            "traffic_agent": 0.15,
        }
        
        self.current_step = 0
        self.max_steps = 50
        self.previous_algorithm = None

    def _get_security_obs(self, obs):
        """Security agent: security_bits, nist_level, threat_level"""
        return np.array([
            obs[1],
            obs[1] / 256,
            np.random.random() * 0.5
        ], dtype=np.float32)
    
    def _get_performance_obs(self, obs):
        """Performance agent: latency, throughput, queue_length"""
        return np.array([
            1 - obs[4],
            obs[5],
            np.random.random() * 0.3
        ], dtype=np.float32)
    
    def _get_resource_obs(self, obs):
        """Resource agent: CPU, memory"""
        return np.array([obs[2], obs[3]], dtype=np.float32)
    
    def _get_traffic_obs(self, obs):
        """Traffic agent: users, workload_phase"""
        workload = obs[6] if len(obs) > 6 else 0
        return np.array([obs[0], workload], dtype=np.float32)
    
    def _get_coordinator_obs(self, obs):
        """Coordinator: combined view"""
        full = np.zeros(10, dtype=np.float32)
        full[:min(7, len(obs))] = obs[:7]
        return full
    
    def _get_specialized_rewards(self, info):
        """Specialized rewards for each agent"""
        
        security_reward = info.get('security_bits', 128) / 256.0
        
        latency = min(1, info.get('latency', 100) / 200)
        throughput = min(1, info.get('throughput', 200) / 500)
        performance_reward = (1 - latency) * 0.5 + throughput * 0.5
        
        cpu = min(1, info.get('cpu', 1000) / 10000)
        memory = min(1, info.get('memory', 200) / 2000)
        resource_reward = (1 - cpu) * 0.5 + (1 - memory) * 0.5
        
        traffic_reward = 1.0 - abs(info.get('workload_phase', 0) - 0.5)
        
        switch_penalty = 0.03 * (info.get('switch_time', 0) / 30)
        
        coordinator_reward = (
            self.coordinator_weights["security_agent"] * security_reward +
            self.coordinator_weights["performance_agent"] * performance_reward +
            self.coordinator_weights["resource_agent"] * resource_reward +
            self.coordinator_weights["traffic_agent"] * traffic_reward -
            switch_penalty
        )
        
        return {
            "security_agent": float(security_reward),
            "performance_agent": float(performance_reward),
            "resource_agent": float(resource_reward),
            "traffic_agent": float(traffic_reward),
            "coordinator": float(coordinator_reward),
        }
    
    def _coordinator_decision(self, actions):
        """Weighted voting by coordinator"""
        votes = np.zeros(self.base_env.action_space.n)
        
        for agent, weight in self.coordinator_weights.items():
            action = actions.get(agent, 0)
            if isinstance(action, (tuple, list, np.ndarray)):
                action = int(action[0]) if len(action) > 0 else 0
            else:
                action = int(action)
            action = max(0, min(action, self.base_env.action_space.n - 1))
            votes[action] += weight
        
        coord_action = actions.get("coordinator", 0)
        if isinstance(coord_action, (tuple, list, np.ndarray)):
            coord_action = int(coord_action[0]) if len(coord_action) > 0 else 0
        else:
            coord_action = int(coord_action)
        coord_action = max(0, min(coord_action, self.base_env.action_space.n - 1))
        votes[coord_action] += 0.2
        
        return int(np.argmax(votes))
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        obs, _ = self.base_env.reset()
        self.current_step = 0
        self.previous_algorithm = None
        
        observations = {
            "security_agent": self._get_security_obs(obs),
            "performance_agent": self._get_performance_obs(obs),
            "resource_agent": self._get_resource_obs(obs),
            "traffic_agent": self._get_traffic_obs(obs),
            "coordinator": self._get_coordinator_obs(obs),
        }
        
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions):
        """Execute step with multi-agent coordination"""
        
        final_action = self._coordinator_decision(actions)
        
        obs, reward, terminated, truncated, info = self.base_env.step(final_action)
        
        specialized_rewards = self._get_specialized_rewards(info)
        
        observations = {
            "security_agent": self._get_security_obs(obs),
            "performance_agent": self._get_performance_obs(obs),
            "resource_agent": self._get_resource_obs(obs),
            "traffic_agent": self._get_traffic_obs(obs),
            "coordinator": self._get_coordinator_obs(obs),
        }
        
        self.current_step += 1
        terminated = terminated or self.current_step >= self.max_steps
        
        infos = {agent: info for agent in self.agents}
        
        return observations, specialized_rewards, terminated, truncated, infos
    
    def render(self):
        pass
    
    def close(self):
        self.base_env.close()
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]