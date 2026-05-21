from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import pandas as pd
import time

from pqc_env import PQCEnv


################################################
# PPO Callback for Logging
################################################

class PPOCallback(BaseCallback):
    def __init__(self, log_file, verbose=0):
        super().__init__(verbose)
        self.log_file = log_file
        self.episode_count = 0
        self.episode_reward = 0
        self.episode_metrics = {}
        self.step_count = 0
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("episode,cumulative_reward,avg_step_reward,latency,throughput,cpu,memory,"
                    "security_bits,switch_time,service_interrupt,jwt_continuity,steps\n")
    
    def _on_step(self):
        rewards = self.locals.get('rewards', [0])
        infos = self.locals.get('infos', [{}])
        
        if rewards and infos:
            self.episode_reward += rewards[0]
            self.step_count += 1
            
            # Track switch metrics from each step
            info = infos[0]
            if info:
                if 'switch_time' not in self.episode_metrics:
                    # Initialize metrics dict
                    self.episode_metrics = {
                        'latency': info.get('latency', 0),
                        'throughput': info.get('throughput', 0),
                        'cpu': info.get('cpu', 0),
                        'memory': info.get('memory', 0),
                        'security_bits': info.get('security_bits', 0),
                        'switch_time': 0,
                        'service_interrupt': 0,
                        'jwt_continuity': info.get('jwt_continuity', 100)
                    }
                
                # Accumulate switch metrics (can be non-zero on any step)
                self.episode_metrics['switch_time'] += info.get('switch_time', 0)
                self.episode_metrics['service_interrupt'] += info.get('service_interrupt', 0)
                self.episode_metrics['jwt_continuity'] = info.get('jwt_continuity', 
                                                                   self.episode_metrics['jwt_continuity'])
            
            dones = self.locals.get('dones', [False])
            if dones[0]:
                self.episode_count += 1
                avg_reward = self.episode_reward / self.step_count if self.step_count > 0 else 0
                
                with open(self.log_file, 'a') as f:
                    f.write(f"{self.episode_count},{self.episode_reward:.4f},{avg_reward:.4f},"
                           f"{self.episode_metrics.get('latency', 0):.2f},"
                           f"{self.episode_metrics.get('throughput', 0):.2f},"
                           f"{self.episode_metrics.get('cpu', 0):.2f},"
                           f"{self.episode_metrics.get('memory', 0):.2f},"
                           f"{self.episode_metrics.get('security_bits', 0)},"
                           f"{self.episode_metrics.get('switch_time', 0):.2f},"
                           f"{self.episode_metrics.get('service_interrupt', 0):.2f},"
                           f"{self.episode_metrics.get('jwt_continuity', 100):.2f},"
                           f"{self.step_count}\n")
                
                if self.verbose > 0:
                    print(f"Episode {self.episode_count}: Reward={self.episode_reward:.4f}, "
                          f"Switch Time={self.episode_metrics.get('switch_time', 0):.2f}ms")
                
                self.episode_reward = 0
                self.episode_metrics = {}
                self.step_count = 0
        
        return True


################################################
# Evaluation Function (Fixed - tracks all steps)
################################################

def evaluate_ppo(model, env, n_episodes=10):
    """Evaluate PPO model with proper switch tracking across all steps"""
    eval_rewards = []
    eval_metrics = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        # Track metrics across episode
        episode_metrics = {}
        total_switch_time = 0
        total_service_interrupt = 0
        final_jwt = 100
        switch_count = 0
        
        while not done and step_count < env.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            
            # Store static metrics from first step
            if not episode_metrics:
                episode_metrics = {
                    'latency': info.get('latency', 0),
                    'throughput': info.get('throughput', 0),
                    'cpu': info.get('cpu', 0),
                    'memory': info.get('memory', 0),
                    'security_bits': info.get('security_bits', 0),
                }
            
            # Accumulate switch metrics (these can be non-zero on any step)
            total_switch_time += info.get('switch_time', 0)
            total_service_interrupt += info.get('service_interrupt', 0)
            if info.get('switch_time', 0) > 0:
                switch_count += 1
            final_jwt = info.get('jwt_continuity', final_jwt)
        
        eval_rewards.append(episode_reward)
        episode_metrics['switch_time'] = total_switch_time
        episode_metrics['service_interrupt'] = total_service_interrupt
        episode_metrics['jwt_continuity'] = final_jwt
        eval_metrics.append(episode_metrics)
        
        print(f"  Episode {episode+1:3d}/{n_episodes}: "
              f"Reward={episode_reward:8.4f} | "
              f"Switches={switch_count:2d} | "
              f"Switch Time={total_switch_time:6.2f}ms | "
              f"JWT={final_jwt:5.1f}%")
    
    return eval_rewards, eval_metrics


################################################
# Main Training
################################################

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    env = Monitor(PQCEnv())
    callback = PPOCallback("results/ppo_results.csv", verbose=1)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=0,
        device="auto"
    )
    
    print("\n" + "="*80)
    print("PPO TRAINING - 10,000 TIMESTEPS")
    print("="*80)
    print(f"\nHyperparameters:")
    print(f"  Total Timesteps: 10,000")
    print(f"  Learning Rate: 3e-4")
    print(f"  Entropy Coef: 0.05 (increased for exploration)")
    print("\n" + "-"*80 + "\n")
    
    start_time = time.time()
    model.learn(total_timesteps=10000, callback=callback, progress_bar=True)
    elapsed = time.time() - start_time
    
    print(f"\nTraining completed in {elapsed:.1f} seconds")
    
    # Save model
    model.save("models/ppo_final")
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION (20 episodes)")
    print("="*80 + "\n")
    
    final_rewards, final_metrics = evaluate_ppo(model, PQCEnv(), n_episodes=20)
    
    print("\n" + "="*80)
    print("FINAL RESULTS - PPO (10,000 timesteps)")
    print("="*80)
    print(f"\nOver 20 evaluation episodes:")
    print(f"  Cumulative Reward: {np.mean(final_rewards):.4f} ± {np.std(final_rewards):.4f}")
    print(f"  Avg Switch Time: {np.mean([m.get('switch_time', 0) for m in final_metrics]):.2f}ms")
    print(f"  Avg JWT: {np.mean([m.get('jwt_continuity', 100) for m in final_metrics]):.2f}%")
    
    print("\nPPO training completed!")
    print("Results saved to: results/ppo_results.csv")


if __name__ == "__main__":
    main()