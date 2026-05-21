# train_mara_jwt.py - TRUE MULTI-AGENT TRAINING WITH PROPER LOGGING
import os
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim

from mara_jwt_env import MARAJWTEnv


class PolicyNetwork(nn.Module):
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
    def __init__(self, total_obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x)


class MARAPPO:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, clip_range=0.2):
        self.env = env
        self.gamma = gamma
        self.clip_range = clip_range
        
        self.policies = {}
        self.optimizers = {}
        
        total_obs_dim = 0
        for agent in env.agents:
            obs_dim = env.observation_space(agent).shape[0]
            action_dim = env.action_space(agent).n
            self.policies[agent] = PolicyNetwork(obs_dim, action_dim)
            self.optimizers[agent] = optim.Adam(self.policies[agent].parameters(), lr=learning_rate)
            total_obs_dim += obs_dim
        
        self.value_net = ValueNetwork(total_obs_dim)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        print(f"\n✅ Created {len(self.policies)} independent policies")
    
    def get_actions(self, observations, deterministic=False):
        actions = {}
        log_probs = {}
        
        for agent in self.env.agents:
            obs = torch.FloatTensor(observations[agent]).unsqueeze(0)
            action_probs = self.policies[agent](obs)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=1).item()
                log_prob = 0
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action).item()
            
            actions[agent] = action
            log_probs[agent] = log_prob
        
        return actions, log_probs
    
    def compute_advantages(self, rewards, dones):
        advantages = []
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                advantage = 0
            advantage = rewards[t] + self.gamma * advantage
            advantages.insert(0, advantage)
        
        advantages = np.array(advantages)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def train_step(self, trajectories):
        for agent in self.env.agents:
            if len(trajectories[agent]['states']) == 0:
                continue
            
            states = torch.FloatTensor(np.array(trajectories[agent]['states']))
            actions = torch.LongTensor(trajectories[agent]['actions'])
            old_log_probs = torch.FloatTensor(trajectories[agent]['log_probs'])
            advantages = torch.FloatTensor(trajectories[agent]['advantages'])
            
            action_probs = self.policies[agent](states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            
            self.optimizers[agent].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policies[agent].parameters(), 0.5)
            self.optimizers[agent].step()
    
    def train(self, total_timesteps=10000, log_file=None):
        episode_rewards = []
        episode = 0
        timestep = 0
        
        # Initialize log file with same format as baselines
        if log_file:
            with open(log_file, 'w') as f:
                f.write("episode,cumulative_reward,avg_step_reward,latency,throughput,cpu,memory,"
                        "security_bits,switch_time,service_interrupt,jwt_continuity,steps\n")
        
        print("\n" + "="*80)
        print("MARA-JWT: TRUE MULTI-AGENT TRAINING")
        print("="*80)
        print(f"\nTotal timesteps: {total_timesteps}")
        print("Logging in same format as baselines (Q, PPO, A2C)")
        print("\n" + "-"*80 + "\n")
        
        while timestep < total_timesteps:
            episode += 1
            observations, _ = self.env.reset()
            done = False
            episode_reward = 0
            step = 0
            
            # Track metrics for this episode
            episode_metrics = {}
            total_switch_time = 0
            total_service_interrupt = 0
            final_jwt = 100
            
            trajectories = {
                agent: {'states': [], 'actions': [], 'log_probs': [], 'rewards': []}
                for agent in self.env.agents
            }
            
            while not done and step < self.env.max_steps and timestep < total_timesteps:
                actions, log_probs = self.get_actions(observations, deterministic=False)
                next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                done = terminated or truncated
                
                # Store metrics from first step
                if not episode_metrics and "coordinator" in infos:
                    info = infos["coordinator"]
                    episode_metrics = {
                        'latency': info.get('latency', 0),
                        'throughput': info.get('throughput', 0),
                        'cpu': info.get('cpu', 0),
                        'memory': info.get('memory', 0),
                        'security_bits': info.get('security_bits', 0),
                    }
                
                # Accumulate switch metrics
                if "coordinator" in infos:
                    info = infos["coordinator"]
                    total_switch_time += info.get('switch_time', 0)
                    total_service_interrupt += info.get('service_interrupt', 0)
                    final_jwt = info.get('jwt_continuity', final_jwt)
                
                for agent in self.env.agents:
                    trajectories[agent]['states'].append(observations[agent])
                    trajectories[agent]['actions'].append(actions[agent])
                    trajectories[agent]['log_probs'].append(log_probs[agent])
                    trajectories[agent]['rewards'].append(rewards["coordinator"])
                
                episode_reward += rewards["coordinator"]
                observations = next_observations
                step += 1
                timestep += 1
            
            episode_rewards.append(episode_reward)
            
            # Log episode in same format as baselines
            if log_file:
                avg_reward = episode_reward / step if step > 0 else 0
                with open(log_file, 'a') as f:
                    f.write(f"{episode},{episode_reward:.4f},{avg_reward:.4f},"
                           f"{episode_metrics.get('latency', 0):.2f},"
                           f"{episode_metrics.get('throughput', 0):.2f},"
                           f"{episode_metrics.get('cpu', 0):.2f},"
                           f"{episode_metrics.get('memory', 0):.2f},"
                           f"{episode_metrics.get('security_bits', 0)},"
                           f"{total_switch_time:.2f},{total_service_interrupt:.2f},{final_jwt:.2f},{step}\n")
            
            # Compute advantages and update
            for agent in self.env.agents:
                advantages = self.compute_advantages(
                    trajectories[agent]['rewards'], 
                    [False] * (len(trajectories[agent]['rewards']) - 1) + [True]
                )
                trajectories[agent]['advantages'] = advantages
            
            self.train_step(trajectories)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode:4d} | Steps: {timestep:6d}/10000 | "
                      f"Reward: {episode_reward:7.2f} | Avg: {avg_reward:7.2f} | "
                      f"Switch Time: {total_switch_time:.2f}ms")
        
        return episode_rewards
    
    def evaluate(self, n_episodes=20):
        eval_rewards = []
        
        print(f"\nEvaluating over {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            observations, _ = self.env.reset()
            done = False
            episode_reward = 0
            step = 0
            
            while not done and step < self.env.max_steps:
                actions, _ = self.get_actions(observations, deterministic=True)
                observations, rewards, terminated, truncated, _ = self.env.step(actions)
                done = terminated or truncated
                episode_reward += rewards["coordinator"]
                step += 1
            
            eval_rewards.append(episode_reward)
            print(f"  Episode {episode+1:3d}/{n_episodes}: Reward={episode_reward:8.2f}")
        
        return eval_rewards
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for agent, policy in self.policies.items():
            torch.save(policy.state_dict(), f"{path}/{agent}_policy.pt")
        print(f"✅ Models saved to {path}/")


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    env = MARAJWTEnv()
    
    trainer = MARAPPO(
        env,
        learning_rate=3e-4,
        gamma=0.99,
        clip_range=0.2
    )
    
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION (SAME AS BASELINES)")
    print("="*80)
    print(f"  Total Timesteps: 10,000")
    print(f"  Learning Rate: 3e-4")
    print(f"  Gamma: 0.99")
    print(f"  Clip Range: 0.2")
    print(f"  Output Format: Same as Q, PPO, A2C")
    print("\n" + "-"*80 + "\n")
    
    start_time = time.time()
    training_rewards = trainer.train(total_timesteps=10000, log_file="results/mara_jwt_results.csv")
    elapsed = time.time() - start_time
    
    print(f"\n✅ Training completed in {elapsed:.1f} seconds")
    
    trainer.save("models/mara_jwt")
    
    print("\n" + "="*80)
    print("FINAL EVALUATION (20 episodes)")
    print("="*80 + "\n")
    
    final_rewards = trainer.evaluate(n_episodes=20)
    
    print("\n" + "="*80)
    print("FINAL RESULTS - MARA-JWT (TRUE MULTI-AGENT)")
    print("="*80)
    print(f"\nOver 20 evaluation episodes:")
    print(f"  Mean Reward: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
    print(f"  Total Reward: {np.sum(final_rewards):.2f}")
    
    print("\n✅ MARA-JWT completed!")
    print("Results saved to: results/mara_jwt_results.csv (same format as baselines)")


if __name__ == "__main__":
    main()