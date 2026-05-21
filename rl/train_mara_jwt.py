# train_mara_jwt.py
import os
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

from mara_jwt_env import MARAJWTEnv


class PolicyNetwork(nn.Module):
    """Policy network for each agent"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    """Value network for advantage calculation"""
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x)


class MARAPPO:
    """
    Multi-Agent PPO for MARA-JWT
    Each agent has its own policy network
    """
    
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_range=0.2, ent_coef=0.01, epochs=10):
        
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.epochs = epochs
        
        # Create separate policy for each agent (TRUE MULTI-AGENT)
        self.policies = {}
        self.optimizers = {}
        
        for agent in env.agents:
            obs_dim = env.observation_space(agent).shape[0]
            action_dim = env.action_space(agent).n
            self.policies[agent] = PolicyNetwork(obs_dim, action_dim)
            self.optimizers[agent] = optim.Adam(self.policies[agent].parameters(), lr=learning_rate)
        
        # Shared value network
        total_obs_dim = sum(env.observation_space(agent).shape[0] for agent in env.agents)
        self.value_net = ValueNetwork(total_obs_dim)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
    
    def get_actions(self, observations):
        """Get actions from all agents' policies"""
        actions = {}
        log_probs = {}
        
        for agent in self.env.agents:
            obs = torch.FloatTensor(observations[agent]).unsqueeze(0)
            action_probs = self.policies[agent](obs)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            actions[agent] = action.item()
            log_probs[agent] = log_prob.item()
        
        return actions, log_probs
    
    def compute_advantages(self, rewards, values, dones):
        """Compute GAE advantages"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def train(self, total_timesteps=10000):
        """Train multi-agent system"""
        
        print("\n" + "="*80)
        print("MARA-JWT: TRAINING TRUE MULTI-AGENT SYSTEM")
        print("="*80)
        print("\nMulti-Agent Architecture:")
        print("  ✓ Each agent has INDEPENDENT policy network")
        print("  ✓ Shared value network for coordination")
        print("  ✓ Weighted voting by coordinator")
        print("  ✓ Specialized observations per agent")
        print("\n" + "-"*80 + "\n")
        
        episode_rewards = []
        episode = 0
        timestep = 0
        
        while timestep < total_timesteps:
            episode += 1
            observations, _ = self.env.reset()
            done = False
            episode_reward = 0
            step = 0
            
            # Storage for episode data
            episode_obs = {agent: [] for agent in self.env.agents}
            episode_actions = {agent: [] for agent in self.env.agents}
            episode_log_probs = {agent: [] for agent in self.env.agents}
            episode_rewards = []
            episode_values = []
            episode_dones = []
            
            while not done and step < self.env.max_steps:
                # Get actions from all agents
                actions, log_probs = self.get_actions(observations)
                
                # Take step in environment
                next_observations, rewards, terminated, truncated, _ = self.env.step(actions)
                done = terminated or truncated
                
                # Get value estimate
                total_obs = np.concatenate([observations[agent] for agent in self.env.agents])
                value = self.value_net(torch.FloatTensor(total_obs).unsqueeze(0)).item()
                
                # Store episode data
                for agent in self.env.agents:
                    episode_obs[agent].append(observations[agent])
                    episode_actions[agent].append(actions[agent])
                    episode_log_probs[agent].append(log_probs[agent])
                
                episode_rewards.append(rewards["coordinator"])
                episode_values.append(value)
                episode_dones.append(done)
                
                episode_reward += rewards["coordinator"]
                observations = next_observations
                step += 1
                timestep += 1
                
                if timestep >= total_timesteps:
                    break
            
            episode_rewards.append(episode_reward)
            
            # Compute advantages
            advantages = self.compute_advantages(episode_rewards, episode_values, episode_dones)
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
            # Update each agent's policy
            for agent in self.env.agents:
                for _ in range(self.epochs):
                    # Convert to tensors
                    obs_tensor = torch.FloatTensor(np.array(episode_obs[agent]))
                    action_tensor = torch.LongTensor(episode_actions[agent])
                    old_log_prob_tensor = torch.FloatTensor(episode_log_probs[agent])
                    adv_tensor = torch.FloatTensor(advantages)
                    
                    # Get new action probabilities
                    action_probs = self.policies[agent](obs_tensor)
                    dist = torch.distributions.Categorical(action_probs)
                    new_log_probs = dist.log_prob(action_tensor)
                    entropy = dist.entropy().mean()
                    
                    # Calculate ratio
                    ratio = torch.exp(new_log_probs - old_log_prob_tensor)
                    
                    # PPO loss
                    surr1 = ratio * adv_tensor
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv_tensor
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Total loss
                    loss = policy_loss - self.ent_coef * entropy
                    
                    # Update policy
                    self.optimizers[agent].zero_grad()
                    loss.backward()
                    self.optimizers[agent].step()
            
            # Update value network
            for _ in range(self.epochs):
                total_obs_list = []
                for i in range(len(episode_rewards)):
                    total_obs = np.concatenate([episode_obs[agent][i] for agent in self.env.agents])
                    total_obs_list.append(total_obs)
                
                obs_tensor = torch.FloatTensor(np.array(total_obs_list))
                returns = torch.FloatTensor(episode_rewards)
                
                values = self.value_net(obs_tensor).squeeze()
                value_loss = nn.MSELoss()(values, returns)
                
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode:4d} | Timesteps: {timestep:6d}/10000 | "
                      f"Avg Reward: {avg_reward:7.2f} | Last Reward: {episode_reward:7.2f}")
        
        return episode_rewards
    
    def save(self, path):
        """Save all policies"""
        for agent, policy in self.policies.items():
            torch.save(policy.state_dict(), f"{path}_{agent}.pt")
        torch.save(self.value_net.state_dict(), f"{path}_value.pt")
        print(f"✅ Models saved to {path}_*.pt")


def evaluate_mara_jwt(agent, env, n_episodes=20):
    """Evaluate trained multi-agent system"""
    eval_rewards = []
    
    for episode in range(n_episodes):
        observations, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done and step_count < env.max_steps:
            actions = {}
            for agent_name in env.agents:
                obs = torch.FloatTensor(observations[agent_name]).unsqueeze(0)
                action_probs = agent.policies[agent_name](obs)
                action = torch.argmax(action_probs, dim=1).item()
                actions[agent_name] = action
            
            observations, rewards, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            episode_reward += rewards["coordinator"]
            step_count += 1
        
        eval_rewards.append(episode_reward)
        print(f"  Episode {episode+1:3d}/{n_episodes}: Reward={episode_reward:8.2f}")
    
    return eval_rewards


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Create environment
    env = MARAJWTEnv()
    
    # Create multi-agent trainer
    trainer = MARAPPO(
        env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        epochs=10
    )
    
    # Train
    start_time = time.time()
    training_rewards = trainer.train(total_timesteps=10000)
    elapsed = time.time() - start_time
    
    print(f"\n✅ Training completed in {elapsed:.1f} seconds")
    
    # Save model
    trainer.save("models/mara_jwt")
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION (20 episodes)")
    print("="*80 + "\n")
    
    final_rewards = evaluate_mara_jwt(trainer, MARAJWTEnv(), n_episodes=20)
    
    # Save results
    results_df = pd.DataFrame({
        'episode': range(1, len(training_rewards) + 1),
        'training_reward': training_rewards,
        'evaluation_reward': final_rewards + [np.nan] * (len(training_rewards) - len(final_rewards))
    })
    results_df.to_csv("results/mara_jwt_results.csv", index=False)
    
    print("\n" + "="*80)
    print("FINAL RESULTS - MARA-JWT (TRUE MULTI-AGENT)")
    print("="*80)
    print(f"\nOver 20 evaluation episodes:")
    print(f"  Mean Reward: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
    print(f"  Total Reward: {np.sum(final_rewards):.2f}")
    print(f"  Min Reward: {np.min(final_rewards):.2f}")
    print(f"  Max Reward: {np.max(final_rewards):.2f}")
    
    print("\n✅ MARA-JWT (True Multi-Agent) training completed!")
    print("Results saved to: results/mara_jwt_results.csv")


if __name__ == "__main__":
    main()