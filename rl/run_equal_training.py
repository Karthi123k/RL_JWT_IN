#!/usr/bin/env python3
"""
Run all algorithms with EQUAL hyperparameters and timesteps
"""

import subprocess
import time
import os

def run_training(script_name, algorithm_name):
    print("\n" + "="*80)
    print(f"TRAINING {algorithm_name}")
    print(f"Timesteps: 10,000 | Episodes: 200 | Episode Length: 50")
    print("="*80 + "\n")
    
    start = time.time()
    result = subprocess.run(["python3", script_name])
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"\n✓ {algorithm_name} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n✗ {algorithm_name} failed")
        return False

def main():
    trainings = [
        ("train_qlearning.py", "Q-LEARNING"),
        ("train_a2c.py", "A2C"),
        ("train_ppo.py", "PPO"),
        ("train_mappo.py", "MAPPO"),
    ]
    
    print("\n" + "="*80)
    print("EQUAL TRAINING FOR ALL ALGORITHMS")
    print("="*80)
    print("\nCommon Settings:")
    print("  Total Timesteps: 10,000")
    print("  Episodes: 200")
    print("  Episode Length: 50 steps")
    print("  Same Network Architecture: [128, 128]")
    print("  Same Random Seeds for reproducibility")
    print("\n" + "-"*80 + "\n")
    
    results = {}
    for script, name in trainings:
        success = run_training(script, name)
        results[name] = "SUCCESS" if success else "FAILED"
        
        if name != trainings[-1][1]:
            print("\nWaiting 5 seconds before next algorithm...")
            time.sleep(5)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("\nResults:")
    for name, status in results.items():
        print(f"  {name}: {status}")
    
    print("\nOutput files saved in results/ directory")

if __name__ == "__main__":
    main()