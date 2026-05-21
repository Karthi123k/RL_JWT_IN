import pandas as pd
import numpy as np
import os

from pqc_env import PQCEnv


################################################
# Load and Process Results
################################################

def load_and_process_results(csv_path, algorithm_name):
    """Load results and compute proper metrics"""
    
    if not os.path.exists(csv_path):
        print(f"⚠️ {algorithm_name} results not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Check if this is episode-level or step-level data
    if 'cumulative_reward' in df.columns:
        # Episode-level data (correct format)
        episode_rewards = df['cumulative_reward'].values
        total_reward = episode_rewards.sum()
        num_episodes = len(episode_rewards)
        
        metrics = {
            'cumulative_reward': total_reward,
            'runtime_adaptability': (episode_rewards > 0).mean() * 100,
            'jwt_continuity': df['jwt_continuity'].mean(),
            'switch_time': df['switch_time'].mean(),
            'service_interrupt': df['service_interrupt'].mean(),
            'latency': df['latency'].mean(),
            'throughput': df['throughput'].mean(),
            'cpu': df['cpu'].mean(),
            'memory': df['memory'].mean(),
            'security_satisfaction': (df['security_bits'] / 256).mean() * 100
        }
        
        print(f"✓ {algorithm_name}: {num_episodes} episodes, Total Reward={total_reward:.2f}")
        
    elif 'reward' in df.columns:
        # Step-level data (need to aggregate)
        print(f"⚠️ {algorithm_name} has step-level data, aggregating...")
        
        # Assume 50 steps per episode
        steps_per_episode = 50
        num_episodes = len(df) // steps_per_episode
        
        # Group by episode
        episode_rewards = []
        episode_metrics = {'switch_time': [], 'service_interrupt': [], 'jwt_continuity': []}
        
        for i in range(num_episodes):
            episode_df = df.iloc[i*steps_per_episode:(i+1)*steps_per_episode]
            episode_rewards.append(episode_df['reward'].sum())
            episode_metrics['switch_time'].append(episode_df['switch_time'].sum())
            episode_metrics['service_interrupt'].append(episode_df['service_interrupt'].sum())
            episode_metrics['jwt_continuity'].append(episode_df['jwt_continuity'].iloc[-1])
        
        metrics = {
            'cumulative_reward': sum(episode_rewards),
            'runtime_adaptability': np.mean([r > 0 for r in episode_rewards]) * 100,
            'jwt_continuity': np.mean(episode_metrics['jwt_continuity']),
            'switch_time': np.mean(episode_metrics['switch_time']),
            'service_interrupt': np.mean(episode_metrics['service_interrupt']),
            'latency': df.groupby(df.index // steps_per_episode)['latency'].first().mean(),
            'throughput': df.groupby(df.index // steps_per_episode)['throughput'].first().mean(),
            'cpu': df.groupby(df.index // steps_per_episode)['cpu'].first().mean(),
            'memory': df.groupby(df.index // steps_per_episode)['memory'].first().mean(),
            'security_satisfaction': (df.groupby(df.index // steps_per_episode)['security_bits'].first().mean() / 256) * 100
        }
        
        print(f"✓ {algorithm_name}: {num_episodes} episodes (aggregated), Total Reward={metrics['cumulative_reward']:.2f}")
    
    else:
        print(f"⚠️ {algorithm_name}: Unknown CSV format")
        return None
    
    return metrics


################################################
# Main Evaluation Function
################################################

def evaluate_all():
    """Evaluate all models from saved results"""
    
    print("\n" + "="*80)
    print("EVALUATING ALL MODELS FROM RESULTS")
    print("="*80 + "\n")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Define result files
    result_files = {
        'Q': "results/qlearning_results.csv",
        'PPO': "results/ppo_results.csv",
        'A2C': "results/a2c_results.csv",
        'MAPPO': "results/mappo_results.csv"
    }
    
    # Load all results
    all_metrics = {}
    
    for algo_name, csv_path in result_files.items():
        print(f"Loading {algo_name}...")
        metrics = load_and_process_results(csv_path, algo_name)
        if metrics:
            all_metrics[algo_name] = metrics
    
    if not all_metrics:
        print("\n⚠️ No result files found. Please run training first.")
        print("\nRun training scripts:")
        print("  python3 train_qlearning_deterministic.py")
        print("  python3 train_ppo.py")
        print("  python3 train_a2c.py")
        print("  python3 train_mappo.py")
        return
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        "Metric": [
            "Cumulative Reward",
            "Runtime Adaptability (%)",
            "JWT Continuity (%)",
            "Switching Time (ms)",
            "Service Interruption (ms)",
            "Latency (ms)",
            "Throughput (rps)",
            "CPU (m)",
            "Memory (MiB)",
            "Security Satisfaction (%)"
        ]
    })
    
    # Add columns for each algorithm
    for algo_name in ['Q', 'PPO', 'A2C', 'MAPPO']:
        if algo_name in all_metrics:
            metrics = all_metrics[algo_name]
            summary[algo_name] = [
                f"{metrics['cumulative_reward']:.2f}",
                f"{metrics['runtime_adaptability']:.2f}",
                f"{metrics['jwt_continuity']:.2f}",
                f"{metrics['switch_time']:.2f}",
                f"{metrics['service_interrupt']:.2f}",
                f"{metrics['latency']:.2f}",
                f"{metrics['throughput']:.2f}",
                f"{metrics['cpu']:.2f}",
                f"{metrics['memory']:.2f}",
                f"{metrics['security_satisfaction']:.2f}"
            ]
        else:
            summary[algo_name] = ["N/A"] * 10
    
    # Print results
    print("\n" + "="*80)
    print("FINAL COMPARISON TABLE")
    print("="*80)
    print("\n")
    print(summary.to_string(index=False))
    
    # Save to CSV
    summary.to_csv("results/final_comparison.csv", index=False)
    print("\n" + "="*80)
    print(f"✓ Results saved to: results/final_comparison.csv")
    
    # Print formatted table for paper
    print("\n" + "="*80)
    print("TABLE FOR JOURNAL PAPER")
    print("="*80)
    print("\n")
    
    # Create LaTeX table format
    available_algos = [a for a in ['Q', 'PPO', 'A2C', 'MAPPO'] if a in all_metrics]
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Performance Comparison of RL Algorithms for PQC Selection}")
    columns = "l" + "c" * len(available_algos)
    print(f"\\begin{{tabular}}{{{columns}}}")
    print("\\hline")
    
    # Header
    header = "Metric & " + " & ".join(available_algos) + " \\\\"
    print(header)
    print("\\hline")
    
    # Rows
    for idx, row in summary.iterrows():
        metric = row['Metric']
        values = [str(row[algo]) for algo in available_algos]
        line = f"{metric} & " + " & ".join(values) + " \\\\"
        print(line)
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)
    
    return summary, all_metrics


################################################
# Run Evaluation
################################################

if __name__ == "__main__":
    summary, results = evaluate_all()