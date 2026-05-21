from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import time

from pqc_env import PQCEnv


class A2CCallback(BaseCallback):
    def __init__(self, log_file, verbose=0):
        super().__init__(verbose)
        self.log_file = log_file
        self.episode_count = 0
        self.episode_reward = 0
        self.episode_metrics = {}
        self.step_count = 0
        
        with open(self.log_file, 'w') as f:
            f.write("episode,cumulative_reward,avg_step_reward,latency,throughput,cpu,memory,"
                    "security_bits,switch_time,service_interrupt,jwt_continuity,steps\n")
    
    def _on_step(self):
        rewards = self.locals.get('rewards', [0])
        infos = self.locals.get('infos', [{}])
        
        if rewards and infos:
            self.episode_reward += rewards[0]
            self.step_count += 1
            
            info = infos[0]
            if info:
                if 'switch_time' not in self.episode_metrics:
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
                    print(f"Episode {self.episode_count}: Reward={self.episode_reward:.2f}")
                
                self.episode_reward = 0
                self.episode_metrics = {}
                self.step_count = 0
        
        return True


def evaluate_a2c(model, env, n_episodes=20):
    eval_rewards = []
    eval_metrics = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
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
            
            if not episode_metrics:
                episode_metrics = {
                    'latency': info.get('latency', 0),
                    'throughput': info.get('throughput', 0),
                    'cpu': info.get('cpu', 0),
                    'memory': info.get('memory', 0),
                    'security_bits': info.get('security_bits', 0),
                }
            
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
              f"Reward={episode_reward:8.2f} | "
              f"Switches={switch_count:2d} | "
              f"Switch Time={total_switch_time:6.2f}ms")
    
    return eval_rewards, eval_metrics


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    env = Monitor(PQCEnv())
    callback = A2CCallback("results/a2c_results.csv", verbose=1)
    
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=64,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=0,
        device="auto"
    )
    
    print("\n" + "="*80)
    print("A2C TRAINING - 10,000 TIMESTEPS")
    print("="*80)
    print(f"\nLearning Rate: 3e-4 | Gamma: 0.99 | Entropy Coef: 0.01")
    print(f"n_steps: 64")
    print("\n" + "-"*80 + "\n")
    
    start_time = time.time()
    model.learn(total_timesteps=10000, callback=callback, progress_bar=True)
    elapsed = time.time() - start_time
    
    print(f"\nTraining completed in {elapsed:.1f} seconds")
    
    model.save("models/a2c_final")
    
    print("\n" + "="*80)
    print("FINAL EVALUATION (20 episodes)")
    print("="*80 + "\n")
    
    final_rewards, final_metrics = evaluate_a2c(model, PQCEnv(), n_episodes=20)
    
    print("\n" + "="*80)
    print("FINAL RESULTS - A2C")
    print("="*80)
    print(f"\nOver 20 evaluation episodes:")
    print(f"  Mean Reward: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
    print(f"  Total Reward: {np.sum(final_rewards):.2f}")
    
    print("\n✅ A2C completed!")


if __name__ == "__main__":
    main()