"""
Deterministic Q-Learning for PQC Algorithm Selection
Journal Paper Implementation - No exploration during evaluation
"""

import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import os
import time
import matplotlib.pyplot as plt

from pqc_env import PQCEnv


################################################
# Q-Learning Agent
################################################

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Q-Learning Agent for PQC algorithm selection
        
        Args:
            action_space: Gym action space
            learning_rate: Alpha (step size)
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum exploration rate
        """
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        self.n_actions = action_space.n
    
    def get_action(self, state, training=True):
        """
        Get action from policy
        
        Args:
            state: Current observation
            training: Whether in training mode (uses epsilon-greedy)
        
        Returns:
            action: Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Exploration during training only
            return self.action_space.sample()
        else:
            # Deterministic policy for evaluation
            state_key = tuple(np.round(state, 2))
            return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-table using Q-learning update rule
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        state_key = tuple(np.round(state, 2))
        next_state_key = tuple(np.round(next_state, 2))
        
        # Q-learning update
        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.gamma * self.q_table[next_state_key][best_next_action] * (1 - done)
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.lr * td_error
        
        # Decay epsilon during training
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Save Q-table to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table saved to {filepath}")
    
    def load(self, filepath):
        """Load Q-table from file"""
        with open(filepath, 'rb') as f:
            self.q_table.update(pickle.load(f))
        print(f"Q-table loaded from {filepath}")


################################################
# Deterministic Evaluation Function (JOURNAL PAPER)
################################################

def evaluate_qlearning_deterministic(agent, env, n_episodes=20, verbose=True):
    """
    DETERMINISTIC evaluation - NO exploration, pure greedy policy
    This is the standard for journal paper comparisons
    
    Args:
        agent: Trained Q-learning agent
        env: PQC Environment
        n_episodes: Number of evaluation episodes
        verbose: Print progress
    
    Returns:
        rewards: List of cumulative rewards per episode
        metrics: List of metric dictionaries per episode
    """
    eval_rewards = []
    eval_metrics = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        episode_metrics = {}
        
        # Track switch metrics across episode
        total_switch_time = 0
        total_service_interrupt = 0
        final_jwt = 100
        switch_count = 0
        action_history = []
        
        while not done and step_count < env.max_steps:
            # DETERMINISTIC action selection (NO exploration)
            state_key = tuple(np.round(obs, 2))
            action = np.argmax(agent.q_table[state_key])
            action_history.append(action)
            
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
            
            # Accumulate switch metrics across ALL steps
            total_switch_time += info.get('switch_time', 0)
            total_service_interrupt += info.get('service_interrupt', 0)
            if info.get('switch_time', 0) > 0:
                switch_count += 1
            final_jwt = info.get('jwt_continuity', final_jwt)
        
        eval_rewards.append(episode_reward)
        episode_metrics['switch_time'] = total_switch_time
        episode_metrics['service_interrupt'] = total_service_interrupt
        episode_metrics['jwt_continuity'] = final_jwt
        episode_metrics['switch_count'] = switch_count
        episode_metrics['unique_actions'] = len(set(action_history))
        eval_metrics.append(episode_metrics)
        
        if verbose:
            print(f"  Episode {episode+1:3d}/{n_episodes}: "
                  f"Reward={episode_reward:8.4f} | "
                  f"Switches={switch_count:2d} | "
                  f"Unique Actions={len(set(action_history))} | "
                  f"Switch Time={total_switch_time:6.2f}ms | "
                  f"JWT={final_jwt:5.1f}%")
    
    return eval_rewards, eval_metrics


################################################
# Training Function
################################################

def train_qlearning(env, agent, total_episodes=200, eval_interval=20, verbose=True):
    """
    Train Q-learning agent
    
    Args:
        env: PQC Environment
        agent: Q-learning agent
        total_episodes: Number of training episodes (200 = 10,000 timesteps)
        eval_interval: Evaluate every N episodes
        verbose: Print progress
    
    Returns:
        training_history: List of training rewards
    """
    training_rewards = []
    training_switches = []
    best_reward = -float("inf")
    
    log_file = "results/qlearning_training.csv"
    with open(log_file, 'w') as f:
        f.write("episode,reward,steps,epsilon,switches\n")
    
    for episode in range(total_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        switch_count = 0
        
        while not done and step_count < env.max_steps:
            action = agent.get_action(obs, training=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            
            if info.get('switch_time', 0) > 0:
                switch_count += 1
        
        training_rewards.append(episode_reward)
        training_switches.append(switch_count)
        
        # Log training progress
        with open(log_file, 'a') as f:
            f.write(f"{episode+1},{episode_reward:.4f},{step_count},{agent.epsilon:.4f},{switch_count}\n")
        
        if verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean(training_rewards[-10:])
            avg_switches = np.mean(training_switches[-10:])
            print(f"Episode {episode+1:3d}/{total_episodes}: "
                  f"Reward={episode_reward:8.4f} | "
                  f"Avg Reward={avg_reward:8.4f} | "
                  f"Switches={switch_count:2d} | "
                  f"Epsilon={agent.epsilon:.4f}")
        
        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            print(f"\n>>> EVALUATION at episode {episode+1} (Deterministic Policy)")
            eval_rewards, eval_metrics = evaluate_qlearning_deterministic(
                agent, PQCEnv(), n_episodes=5, verbose=False
            )
            mean_reward = np.mean(eval_rewards)
            mean_switch_time = np.mean([m.get('switch_time', 0) for m in eval_metrics])
            
            print(f"    Evaluation Reward: {mean_reward:.4f} ± {np.std(eval_rewards):.4f}")
            print(f"    Avg Switch Time: {mean_switch_time:.2f}ms\n")
            
            if mean_reward > best_reward:
                best_reward = mean_reward
                agent.save("models/qlearning_best.pkl")
    
    return training_rewards, training_switches


################################################
# Plotting Function (for paper)
################################################

def plot_results(training_rewards, training_switches, final_rewards, save_path="results/"):
    """Generate plots for journal paper"""
    
    # Figure 1: Learning Curve
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_rewards, alpha=0.7, label='Episode Reward')
    # Add moving average
    window = 10
    moving_avg = np.convolve(training_rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(training_rewards)), moving_avg, 'r-', label=f'{window}-episode MA')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Q-Learning Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Figure 2: Switching Behavior
    plt.subplot(1, 2, 2)
    plt.plot(training_switches, alpha=0.7, label='Switches per Episode')
    moving_avg_switches = np.convolve(training_switches, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(training_switches)), moving_avg_switches, 'r-', label=f'{window}-episode MA')
    plt.xlabel('Episode')
    plt.ylabel('Number of Algorithm Switches')
    plt.title('Q-Learning Switching Behavior')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/qlearning_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save final results
    print_final_results(final_rewards, save_path)


def print_final_results(final_rewards, save_path):
    """Print and save final results for journal paper"""
    
    results = {
        'mean_reward': np.mean(final_rewards),
        'std_reward': np.std(final_rewards),
        'min_reward': np.min(final_rewards),
        'max_reward': np.max(final_rewards),
        'confidence_interval': 1.96 * np.std(final_rewards) / np.sqrt(len(final_rewards))
    }
    
    print("\n" + "="*80)
    print("FINAL RESULTS - DETERMINISTIC Q-LEARNING")
    print("="*80)
    print(f"\nOver {len(final_rewards)} evaluation episodes (deterministic policy):")
    print(f"  Mean Cumulative Reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"  95% Confidence Interval: [{results['mean_reward'] - results['confidence_interval']:.4f}, "
          f"{results['mean_reward'] + results['confidence_interval']:.4f}]")
    print(f"  Min Reward: {results['min_reward']:.4f}")
    print(f"  Max Reward: {results['max_reward']:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv(f"{save_path}/qlearning_final_stats.csv", index=False)


################################################
# Main Training Script
################################################

def main():
    # Create directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Create environment
    env = PQCEnv()
    
    # Create Q-learning agent
    agent = QLearningAgent(
        action_space=env.action_space,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Training parameters
    TOTAL_EPISODES = 200  # 200 episodes * 50 steps = 10,000 timesteps
    EVAL_INTERVAL = 20
    
    print("\n" + "="*80)
    print("DETERMINISTIC Q-LEARNING FOR JOURNAL PAPER")
    print("="*80)
    print("\nExperimental Setup:")
    print(f"  Total Training Episodes: {TOTAL_EPISODES} (10,000 timesteps)")
    print(f"  Evaluation: Deterministic (greedy policy, NO exploration)")
    print(f"  Number of Evaluation Episodes: 20")
    print("\nHyperparameters:")
    print(f"  Learning Rate (α): 0.1")
    print(f"  Discount Factor (γ): 0.99")
    print(f"  Initial Epsilon (ε): 1.0")
    print(f"  Epsilon Decay: 0.995")
    print(f"  Minimum Epsilon: 0.01")
    print("\n" + "-"*80 + "\n")
    
    start_time = time.time()
    
    # Train agent
    training_rewards, training_switches = train_qlearning(
        env, agent, 
        total_episodes=TOTAL_EPISODES, 
        eval_interval=EVAL_INTERVAL,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds")
    
    # Save final model
    agent.save("models/qlearning_final.pkl")
    
    # FINAL EVALUATION (DETERMINISTIC - for journal paper)
    print("\n" + "="*80)
    print("FINAL DETERMINISTIC EVALUATION (20 episodes)")
    print("="*80)
    print("\nEvaluating with greedy policy (NO exploration)...\n")
    
    final_rewards, final_metrics = evaluate_qlearning_deterministic(
        agent, PQCEnv(), n_episodes=20, verbose=True
    )
    
    # Calculate aggregate metrics
    all_switch_times = [m.get('switch_time', 0) for m in final_metrics]
    all_service_interrupts = [m.get('service_interrupt', 0) for m in final_metrics]
    all_jwt = [m.get('jwt_continuity', 100) for m in final_metrics]
    all_security = [m.get('security_bits', 0) for m in final_metrics]
    all_unique_actions = [m.get('unique_actions', 0) for m in final_metrics]
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY (For Journal Paper)")
    print("="*80)
    print(f"\nPerformance Metrics (20 deterministic episodes):")
    print(f"  Cumulative Reward: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
    print(f"  Algorithm Switches: {np.mean(all_switch_times):.2f} ± {np.std(all_switch_times):.2f} ms")
    print(f"  Service Interruption: {np.mean(all_service_interrupts):.2f} ± {np.std(all_service_interrupts):.2f} ms")
    print(f"  JWT Continuity: {np.mean(all_jwt):.2f} ± {np.std(all_jwt):.2f}%")
    print(f"  Security Satisfaction: {np.mean(all_security):.2f} ± {np.std(all_security):.2f}")
    print(f"  Unique Algorithms Used: {np.mean(all_unique_actions):.2f} ± {np.std(all_unique_actions):.2f}")
    
    # Save all results to CSV
    results_df = pd.DataFrame({
        'episode': range(1, len(final_rewards) + 1),
        'cumulative_reward': final_rewards,
        'avg_step_reward': [r/50 for r in final_rewards],
        'latency': [m.get('latency', 0) for m in final_metrics],
        'throughput': [m.get('throughput', 0) for m in final_metrics],
        'cpu': [m.get('cpu', 0) for m in final_metrics],
        'memory': [m.get('memory', 0) for m in final_metrics],
        'security_bits': all_security,
        'switch_time': all_switch_times,
        'service_interrupt': all_service_interrupts,
        'jwt_continuity': all_jwt,
        'steps': [50] * len(final_rewards),
        'unique_actions': all_unique_actions
    })
    results_df.to_csv("results/qlearning_deterministic_results.csv", index=False)
    
    # Generate plots
    plot_results(training_rewards, training_switches, final_rewards)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)
    print("\nOutput files saved:")
    print("  📊 results/qlearning_deterministic_results.csv - Final evaluation results")
    print("  📊 results/qlearning_training.csv - Training history")
    print("  📊 results/qlearning_final_stats.csv - Summary statistics")
    print("  📊 results/qlearning_results.png - Learning curves")
    print("  🤖 models/qlearning_final.pkl - Trained Q-table")
    print("  🤖 models/qlearning_best.pkl - Best model during training")
    
    print("\n✅ Q-Learning training completed successfully!")
    print("\nCitation for paper:")
    print("  - Evaluation method: Deterministic (greedy) policy")
    print("  - No exploration during evaluation")
    print("  - Results are reproducible with fixed random seed")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    main()