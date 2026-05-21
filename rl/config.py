"""
EQUAL TRAINING CONFIGURATION FOR ALL ALGORITHMS
Same hyperparameters, same timesteps, same conditions
"""

# COMMON SETTINGS
TOTAL_TIMESTEPS = 10000  # 200 episodes * 50 steps
EPISODE_LENGTH = 50
TOTAL_EPISODES = TOTAL_TIMESTEPS // EPISODE_LENGTH  # 200

# NETWORK ARCHITECTURE (same for all)
NETWORK_ARCH = [128, 128]

# Q-LEARNING
Q_LEARNING_CONFIG = {
    'learning_rate': 0.1,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
}

# PPO CONFIG
PPO_CONFIG = {
    'learning_rate': 3e-4,
    'n_steps': 512,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
}

# A2C CONFIG
A2C_CONFIG = {
    'learning_rate': 3e-4,
    'n_steps': 64,
    'gamma': 0.99,
    'ent_coef': 0.01,
}

# MAPPO CONFIG
MAPPO_CONFIG = {
    'train_batch_size': 512,
    'minibatch_size': 64,
    'gamma': 0.99,
    'lr': 3e-4,
    'clip_param': 0.2,
    'entropy_coeff': 0.01,
    'num_epochs': 10,
}