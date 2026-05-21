import os
import ray
import numpy as np
import time
import pandas as pd

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from gymnasium.spaces import Box, Discrete

from mappo_env import PQCMultiAgentEnv


def evaluate_mappo_policy(algo, n_episodes=10, verbose=True):
    """Evaluate MAPPO policy"""
    eval_rewards = []
    eval_metrics = []
    
    for episode in range(n_episodes):
        eval_env = PQCMultiAgentEnv()
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        episode_metrics = {}
        total_switch_time = 0
        total_service_interrupt = 0
        final_jwt = 100
        switch_count = 0
        
        while not done and step_count < eval_env.base_env.max_steps:
            actions = {}
            for agent in eval_env.agents:
                # Get policy for each agent
                policy = algo.get_policy(f"{agent}_policy")
                action = policy.compute_single_action(obs[agent])
                
                if isinstance(action, (tuple, list, np.ndarray)):
                    action = int(action[0]) if len(action) > 0 else 0
                else:
                    action = int(action)
                
                actions[agent] = action
            
            obs, rewards, terminated, truncated, infos = eval_env.step(actions)
            done = terminated["__all__"] or truncated["__all__"]
            episode_reward += rewards["coordinator"]
            step_count += 1
            
            if "coordinator" in infos:
                info = infos["coordinator"]
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
        eval_env.close()
        
        if verbose:
            print(f"  Episode {episode+1:3d}/{n_episodes}: "
                  f"Reward={episode_reward:8.2f} | "
                  f"Switches={switch_count:2d} | "
                  f"Switch Time={total_switch_time:6.2f}ms")
    
    return eval_rewards, eval_metrics


def main():
    # Create directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("models/mappo", exist_ok=True)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Register environment
    register_env("pqc_multi", lambda config: PQCMultiAgentEnv(config))
    
    # Define observation and action spaces for each specialized agent
    security_obs_space = Box(0, 1, (2,), dtype=np.float32)
    performance_obs_space = Box(0, 1, (2,), dtype=np.float32)
    resource_obs_space = Box(0, 1, (2,), dtype=np.float32)
    traffic_obs_space = Box(0, 1, (2,), dtype=np.float32)
    coordinator_obs_space = Box(0, 1, (8,), dtype=np.float32)
    action_space = Discrete(11)  # 11 PQC algorithms
    
    # TRUE MAPPO: Each agent has its OWN policy
    config = (
        PPOConfig()
        .environment(env="pqc_multi")
        .training(
            train_batch_size=1024,
            minibatch_size=128,
            gamma=0.99,
            lr=5e-4,
            clip_param=0.3,
            entropy_coeff=0.05,
            num_epochs=15,
        )
        .multi_agent(
            # EACH AGENT gets its OWN policy with its specific observation space
            policies={
                "security_agent_policy": (None, security_obs_space, action_space, {}),
                "performance_agent_policy": (None, performance_obs_space, action_space, {}),
                "resource_agent_policy": (None, resource_obs_space, action_space, {}),
                "traffic_agent_policy": (None, traffic_obs_space, action_space, {}),
                "coordinator_policy": (None, coordinator_obs_space, action_space, {}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: f"{agent_id}_policy",
            policies_to_train=[
                "security_agent_policy", 
                "performance_agent_policy", 
                "resource_agent_policy", 
                "traffic_agent_policy", 
                "coordinator_policy"
            ],
        )
        .resources(num_gpus=0)
        .framework("torch")
    )
    
    print("\n" + "="*80)
    print("TRUE MARA-JWT MAPPO TRAINING")
    print("="*80)
    print("\nArchitecture:")
    print("  ✓ Security Agent (own policy) - optimizes security")
    print("  ✓ Performance Agent (own policy) - optimizes latency/throughput")
    print("  ✓ Resource Agent (own policy) - optimizes CPU/memory")
    print("  ✓ Traffic Agent (own policy) - optimizes for workload")
    print("  ✓ Coordinator (own policy) - combines all decisions")
    print("\nHyperparameters:")
    print(f"  Train Batch Size: 1024")
    print(f"  Minibatch Size: 128")
    print(f"  Num Epochs: 15")
    print(f"  Learning Rate: 5e-4")
    print(f"  Clip Range: 0.3")
    print(f"  Entropy Coeff: 0.05")
    print("\n" + "-"*80 + "\n")
    
    # Build algorithm
    algo = config.build_algo()
    
    # Training
    TOTAL_TIMESTEPS = 25000
    current_timesteps = 0
    iteration = 0
    best_reward = -float("inf")
    training_rewards = []
    
    print("Training Progress:")
    print("-" * 70)
    start_time = time.time()
    
    while current_timesteps < TOTAL_TIMESTEPS:
        iteration += 1
        result = algo.train()
        
        train_reward = result["env_runners"]["episode_return_mean"]
        new_timesteps = int(result["num_env_steps_sampled_lifetime"])
        current_timesteps = new_timesteps
        training_rewards.append(train_reward)
        elapsed = time.time() - start_time
        
        print(f"Iter {iteration:3d} | Steps: {current_timesteps:6d}/{TOTAL_TIMESTEPS} | "
              f"Reward: {train_reward:7.2f} | Time: {elapsed:5.1f}s")
        
        if train_reward > best_reward:
            best_reward = train_reward
            algo.save(os.path.abspath("./models/mappo/best_checkpoint"))
            print(f"  ✓ New best model! Reward: {best_reward:.2f}")
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION (20 episodes)")
    print("="*80 + "\n")
    
    final_rewards, final_metrics = evaluate_mappo_policy(algo, n_episodes=20, verbose=True)
    
    print("\n" + "="*80)
    print("TRUE MARA-JWT MAPPO FINAL RESULTS")
    print("="*80)
    print(f"\nOver 20 evaluation episodes:")
    print(f"  Mean Reward: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
    print(f"  Total Reward: {np.sum(final_rewards):.2f}")
    print(f"  Min Reward: {np.min(final_rewards):.2f}")
    print(f"  Max Reward: {np.max(final_rewards):.2f}")
    print(f"  Avg Switch Time: {np.mean([m.get('switch_time', 0) for m in final_metrics]):.2f}ms")
    print(f"  Avg JWT: {np.mean([m.get('jwt_continuity', 100) for m in final_metrics]):.2f}%")
    
    # Save model
    algo.save(os.path.abspath("./models/mappo/final_model"))
    print(f"\n✅ True MARA-JWT MAPPO model saved to: models/mappo/final_model")
    
    # Save training history
    history_df = pd.DataFrame({
        'iteration': range(1, len(training_rewards) + 1),
        'train_reward': training_rewards,
        'timesteps': [min(i * 1024, TOTAL_TIMESTEPS) for i in range(1, len(training_rewards) + 1)]
    })
    history_df.to_csv("results/mappo_training_history.csv", index=False)
    print("Training history saved to: results/mappo_training_history.csv")
    
    print("\n✅ MARA-JWT MAPPO training completed!")
    
    ray.shutdown()


if __name__ == "__main__":
    main()