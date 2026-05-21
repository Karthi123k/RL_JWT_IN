import pandas as pd
import numpy as np
import os


def load_results():
    """Load results from all algorithms"""
    results = {}
    
    # Q-learning
    if os.path.exists("results/qlearning_results.csv"):
        df = pd.read_csv("results/qlearning_results.csv")
        if 'cumulative_reward' in df.columns:
            results['Q-learning'] = {
                'total_reward': df['cumulative_reward'].sum(),
                'mean_reward': df['cumulative_reward'].mean(),
                'std_reward': df['cumulative_reward'].std(),
                'switches': df.get('switch_time', pd.Series([0]*len(df))).mean()
            }
        else:
            # Aggregate step-level data
            episodes = len(df) // 50
            rewards = [df.iloc[i*50:(i+1)*50]['reward'].sum() for i in range(episodes)]
            results['Q-learning'] = {
                'total_reward': sum(rewards),
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'switches': df['switch_time'].mean()
            }
    
    # PPO
    if os.path.exists("results/ppo_results.csv"):
        df = pd.read_csv("results/ppo_results.csv")
        if 'cumulative_reward' in df.columns:
            results['PPO'] = {
                'total_reward': df['cumulative_reward'].sum(),
                'mean_reward': df['cumulative_reward'].mean(),
                'std_reward': df['cumulative_reward'].std(),
                'switches': df.get('switch_time', pd.Series([0]*len(df))).mean()
            }
        else:
            episodes = len(df) // 50
            rewards = [df.iloc[i*50:(i+1)*50]['reward'].sum() for i in range(episodes)]
            results['PPO'] = {
                'total_reward': sum(rewards),
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'switches': df['switch_time'].mean()
            }
    
    # A2C
    if os.path.exists("results/a2c_results.csv"):
        df = pd.read_csv("results/a2c_results.csv")
        if 'cumulative_reward' in df.columns:
            results['A2C'] = {
                'total_reward': df['cumulative_reward'].sum(),
                'mean_reward': df['cumulative_reward'].mean(),
                'std_reward': df['cumulative_reward'].std(),
                'switches': df.get('switch_time', pd.Series([0]*len(df))).mean()
            }
        else:
            episodes = len(df) // 50
            rewards = [df.iloc[i*50:(i+1)*50]['reward'].sum() for i in range(episodes)]
            results['A2C'] = {
                'total_reward': sum(rewards),
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'switches': df['switch_time'].mean()
            }
    
    # MAPPO
    if os.path.exists("results/mappo_results.csv"):
        df = pd.read_csv("results/mappo_results.csv")
        results['MAPPO'] = {
            'total_reward': df['cumulative_reward'].sum(),
            'mean_reward': df['cumulative_reward'].mean(),
            'std_reward': df['cumulative_reward'].std(),
            'switches': df.get('switches', pd.Series([0]*len(df))).mean()
        }
    
    return results


def main():
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON (Same Hyperparameters: 10,000 timesteps)")
    print("="*80)
    
    results = load_results()
    
    if not results:
        print("\nNo result files found. Please run training first:")
        print("  python3 train_qlearning.py")
        print("  python3 train_ppo.py")
        print("  python3 train_a2c.py")
        print("  python3 train_mappo.py")
        return
    
    # Create comparison table
    comparison = pd.DataFrame({
        "Algorithm": list(results.keys()),
        "Total Reward (20 episodes)": [results[a]['total_reward'] for a in results.keys()],
        "Mean Reward per Episode": [f"{results[a]['mean_reward']:.2f} ± {results[a]['std_reward']:.2f}" for a in results.keys()],
        "Avg Switch Time (ms)": [f"{results[a]['switches']:.2f}" for a in results.keys()],
    })
    
    print("\n" + comparison.to_string(index=False))
    
    # Find best algorithm
    best_algo = max(results.keys(), key=lambda x: results[x]['total_reward'])
    print(f"\n🏆 Best Algorithm: {best_algo}")
    print(f"   Total Reward: {results[best_algo]['total_reward']:.2f}")
    
    # Save comparison
    comparison.to_csv("results/algorithm_comparison.csv", index=False)
    print("\n✅ Comparison saved to: results/algorithm_comparison.csv")


if __name__ == "__main__":
    main()