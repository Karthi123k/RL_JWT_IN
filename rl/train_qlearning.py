import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import os
import time

from pqc_env import PQCEnv


class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        self.n_actions = action_space.n
    
    def get_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            state_key = tuple(np.round(state, 2))
            return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state, done):
        state_key = tuple(np.round(state, 2))
        next_state_key = tuple(np.round(next_state, 2))
        
        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.gamma * self.q_table[next_state_key][best_next_action] * (1 - done)
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.lr * td_error
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table saved to {filepath}")


def evaluate_qlearning(agent, env, n_episodes=20):
    """Deterministic evaluation"""
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
            action = agent.get_action(obs, training=False)
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
    
    env = PQCEnv()
    log_file = "results/qlearning_results.csv"
    
    # Initialize log file
    with open(log_file, 'w') as f:
        f.write("episode,cumulative_reward,avg_step_reward,latency,throughput,cpu,memory,"
                "security_bits,switch_time,service_interrupt,jwt_continuity,steps\n")
    
    agent = QLearningAgent(
        action_space=env.action_space,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    TOTAL_EPISODES = 200  # 10,000 timesteps
    EVAL_INTERVAL = 20
    
    print("\n" + "="*80)
    print("Q-LEARNING TRAINING - 10,000 TIMESTEPS")
    print("="*80)
    print(f"\nTotal Episodes: {TOTAL_EPISODES} (10,000 timesteps)")
    print(f"Learning Rate: 0.1 | Gamma: 0.99 | Epsilon Decay: 0.995")
    print("\n" + "-"*80 + "\n")
    
    best_reward = -float("inf")
    episode_counter = 0
    start_time = time.time()
    
    for episode in range(TOTAL_EPISODES):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        episode_metrics = {}
        switch_count = 0
        
        while not done and step_count < env.max_steps:
            action = agent.get_action(obs, training=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            
            if not episode_metrics:
                episode_metrics = info
            
            if info.get('switch_time', 0) > 0:
                switch_count += 1
        
        episode_counter += 1
        avg_reward = episode_reward / step_count if step_count > 0 else 0
        
        with open(log_file, 'a') as f:
            f.write(f"{episode_counter},{episode_reward:.4f},{avg_reward:.4f},"
                   f"{episode_metrics.get('latency', 0):.2f},"
                   f"{episode_metrics.get('throughput', 0):.2f},"
                   f"{episode_metrics.get('cpu', 0):.2f},"
                   f"{episode_metrics.get('memory', 0):.2f},"
                   f"{episode_metrics.get('security_bits', 0)},"
                   f"{episode_metrics.get('switch_time', 0):.2f},"
                   f"{episode_metrics.get('service_interrupt', 0):.2f},"
                   f"{episode_metrics.get('jwt_continuity', 100):.2f},"
                   f"{step_count}\n")
        
        if (episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode+1:3d}/{TOTAL_EPISODES}: "
                  f"Reward={episode_reward:8.2f} | "
                  f"Switches={switch_count:2d} | "
                  f"Epsilon={agent.epsilon:.4f} | "
                  f"Time={elapsed:5.1f}s")
        
        if (episode + 1) % EVAL_INTERVAL == 0:
            print(f"\n>>> EVALUATING at episode {episode+1}...")
            eval_rewards, eval_metrics = evaluate_qlearning(agent, PQCEnv(), n_episodes=5)
            mean_reward = np.mean(eval_rewards)
            print(f">>> Evaluation Reward: {mean_reward:.2f} ± {np.std(eval_rewards):.2f}\n")
            
            if mean_reward > best_reward:
                best_reward = mean_reward
                agent.save("models/qlearning_best.pkl")
    
    agent.save("models/qlearning_final.pkl")
    
    print("\n" + "="*80)
    print("FINAL EVALUATION (20 episodes)")
    print("="*80 + "\n")
    
    final_rewards, final_metrics = evaluate_qlearning(agent, PQCEnv(), n_episodes=20)
    
    print("\n" + "="*80)
    print("FINAL RESULTS - Q-LEARNING")
    print("="*80)
    print(f"\nOver 20 evaluation episodes:")
    print(f"  Mean Reward: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
    print(f"  Total Reward: {np.sum(final_rewards):.2f}")
    
    print("\n✅ Q-Learning completed!")


if __name__ == "__main__":
    np.random.seed(42)
    main()