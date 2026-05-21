from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Box, Discrete
from pqc_env import PQCEnv
import numpy as np


class PQCMultiAgentEnv(MultiAgentEnv):
    """
    TRUE MARA-JWT Multi-Agent Environment
    Each agent has specialized observation and reward
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        self.base_env = PQCEnv()
        
        # Define specialized agents
        self.agents = [
            "security_agent",      # Optimizes security
            "performance_agent",   # Optimizes latency/throughput
            "resource_agent",      # Optimizes CPU/Memory
            "traffic_agent",       # Optimizes for workload
            "coordinator"          # Combines all agents' decisions
        ]
        
        self.possible_agents = self.agents.copy()
        
        # DIFFERENT OBSERVATION SPACES for each specialized agent
        self.observation_spaces = {
            "security_agent": Box(0, 1, (2,), dtype=np.float32),     # security_bits, threat_level
            "performance_agent": Box(0, 1, (2,), dtype=np.float32),  # latency, throughput
            "resource_agent": Box(0, 1, (2,), dtype=np.float32),     # cpu, memory
            "traffic_agent": Box(0, 1, (2,), dtype=np.float32),      # users, workload_phase
            "coordinator": Box(0, 1, (8,), dtype=np.float32),        # combined features
        }
        
        # Each agent has SAME action space (PQC algorithms)
        self.action_spaces = {
            agent: self.base_env.action_space
            for agent in self.agents
        }
        
        # Track previous algorithm for migration metrics
        self.previous_algorithm = None
        self.coordinator_weights = np.array([0.35, 0.35, 0.15, 0.15])  # Security, Performance, Resource, Traffic

    def reset(self, *, seed=None, options=None):
        obs, info = self.base_env.reset()
        self.previous_algorithm = None
        
        # Create SPECIALIZED observations for each agent
        observations = {
            "security_agent": self._get_security_obs(obs),
            "performance_agent": self._get_performance_obs(obs),
            "resource_agent": self._get_resource_obs(obs),
            "traffic_agent": self._get_traffic_obs(obs),
            "coordinator": self._get_coordinator_obs(obs),
        }
        
        infos = {agent: info for agent in self.agents}
        return observations, infos
    
    def _get_security_obs(self, obs):
        """Security agent observes: security_bits, threat_level"""
        threat_level = obs[1] / 256  # Normalize
        return np.array([obs[1] / 256, threat_level], dtype=np.float32)
    
    def _get_performance_obs(self, obs):
        """Performance agent observes: latency, throughput"""
        return np.array([obs[4], obs[5]], dtype=np.float32)  # latency, throughput
    
    def _get_resource_obs(self, obs):
        """Resource agent observes: cpu, memory"""
        return np.array([obs[2], obs[3]], dtype=np.float32)  # cpu, memory
    
    def _get_traffic_obs(self, obs):
        """Traffic agent observes: users, workload_phase"""
        workload_phase = obs[6] if len(obs) > 6 else 0
        return np.array([obs[0], workload_phase], dtype=np.float32)  # users, workload
    
    def _get_coordinator_obs(self, obs):
        """Coordinator observes combined features"""
        return np.array(obs[:8] if len(obs) >= 8 else np.pad(obs, (0, 8-len(obs))), dtype=np.float32)
    
    def _get_specialized_rewards(self, info, action_taken):
        """Calculate specialized rewards for each agent"""
        
        # Security agent reward
        security_reward = info.get('security_bits', 128) / 256
        
        # Performance agent reward (negative latency, positive throughput)
        performance_reward = (
            -info.get('latency', 100) / 200 +  # Normalize latency
            info.get('throughput', 200) / 500   # Normalize throughput
        ) / 2
        
        # Resource agent reward (negative CPU, negative memory)
        resource_reward = -(
            info.get('cpu', 1000) / 10000 + 
            info.get('memory', 200) / 2000
        ) / 2
        
        # Traffic agent reward (based on workload adaptation)
        traffic_reward = 1.0 - abs(info.get('workload_phase', 0) - 0.5) * 2
        
        # Coordinator reward (overall system reward)
        coordinator_reward = (
            0.35 * security_reward +
            0.35 * performance_reward +
            0.15 * resource_reward +
            0.15 * traffic_reward -
            0.03 * (info.get('switch_time', 0) / 30)
        )
        
        return {
            "security_agent": security_reward,
            "performance_agent": performance_reward,
            "resource_agent": resource_reward,
            "traffic_agent": traffic_reward,
            "coordinator": coordinator_reward,
        }
    
    def _coordinator_decisions(self, actions):
        """Coordinator combines all agents' decisions"""
        # Get Q-values from each agent's action
        agent_actions = {
            agent: actions.get(agent, 0) for agent in self.agents if agent != "coordinator"
        }
        
        # Weighted voting by coordinator
        final_action = 0
        max_vote = -float("inf")
        
        for alg_idx in range(self.base_env.action_space.n):
            vote = 0
            for i, agent in enumerate(["security_agent", "performance_agent", "resource_agent", "traffic_agent"]):
                agent_action = agent_actions.get(agent, 0)
                if agent_action == alg_idx:
                    vote += self.coordinator_weights[i]
            
            if vote > max_vote:
                max_vote = vote
                final_action = alg_idx
        
        return final_action

    def step(self, actions):
        """
        Execute actions for all agents.
        Coordinator combines all agents' decisions.
        """
        # Coordinator combines all agents' decisions
        final_action = self._coordinator_decisions(actions)
        
        # Take step in base environment
        obs, reward, terminated, truncated, info = self.base_env.step(final_action)
        
        # Get specialized rewards for each agent
        specialized_rewards = self._get_specialized_rewards(info, final_action)
        
        # Create specialized observations for next state
        observations = {
            "security_agent": self._get_security_obs(obs),
            "performance_agent": self._get_performance_obs(obs),
            "resource_agent": self._get_resource_obs(obs),
            "traffic_agent": self._get_traffic_obs(obs),
            "coordinator": self._get_coordinator_obs(obs),
        }
        
        # Each agent gets its specialized reward
        rewards = {
            "security_agent": specialized_rewards["security_agent"],
            "performance_agent": specialized_rewards["performance_agent"],
            "resource_agent": specialized_rewards["resource_agent"],
            "traffic_agent": specialized_rewards["traffic_agent"],
            "coordinator": specialized_rewards["coordinator"],
        }
        
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: info for agent in self.agents}
        
        terminations["__all__"] = terminated
        truncations["__all__"] = truncated
        
        return observations, rewards, terminations, truncations, infos