import gymnasium as gym
import numpy as np
import pandas as pd

from gymnasium.spaces import Box, Discrete


class PQCEnv(gym.Env):

    metadata = {"render_modes": []}

    ################################################
    # INIT
    ################################################

    def __init__(self):

        super().__init__()

        import os
        local_path = os.path.join(os.path.dirname(__file__), "overall_benchmark.csv")
        if os.path.exists(local_path):
            self.df = pd.read_csv(local_path)
        else:
            self.df = pd.read_csv(
                "../locust/logs/reuse/overall_benchmark.csv"
            )

        self.algorithms = sorted(self.df["key_size"].unique())

        ################################################
        # actions
        ################################################
        self.action_space = Discrete(len(self.algorithms))

        ################################################
        # states (7 dimensions with workload phase)
        ################################################
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )

        self.previous_algorithm = None
        self.current_step = 0
        self.max_steps = 50
        
        # Dynamic workload parameters
        self.workload_phase = 0  # 0=normal, 1=high, 2=low
        self.phase_duration = 0
        self.base_users = None
        self.user_multiplier = 1.0
        self._phase_changed = False

    ################################################
    # normalize
    ################################################
    def normalize(self, value, column):
        """Safe normalization with clipping to [0,1]"""
        minv = self.df[column].min()
        maxv = self.df[column].max()
        
        if maxv - minv < 1e-9:
            return 0.5
        
        normalized = (value - minv) / (maxv - minv + 1e-9)
        return float(np.clip(normalized, 0.0, 1.0))

    ################################################
    # get algorithm performance from benchmark data
    ################################################
    def get_algorithm_performance(self, algorithm, users):
        """Get real benchmark data for specific algorithm and user count"""
        candidates = self.df[
            (self.df["key_size"] == algorithm) &
            (self.df["users"] == users)
        ]
        
        if len(candidates) == 0:
            # Find nearest users value
            unique_users = sorted(self.df["users"].unique())
            nearest_users = min(unique_users, key=lambda x: abs(x - users))
            candidates = self.df[
                (self.df["key_size"] == algorithm) &
                (self.df["users"] == nearest_users)
            ]
        
        if len(candidates) == 0:
            candidates = self.df[self.df["key_size"] == algorithm]
        
        return candidates.iloc[0]

    ################################################
    # update workload phase (dynamic)
    ################################################
    def update_workload_phase(self):
        """Dynamically change workload conditions"""
        self.phase_duration += 1
        self._phase_changed = False
        
        # Change phase every 8-15 steps
        if self.phase_duration >= np.random.randint(8, 16):
            self.phase_duration = 0
            
            # Cycle through phases
            self.workload_phase = (self.workload_phase + 1) % 3
            
            # Update user multiplier based on phase
            if self.workload_phase == 0:  # Normal
                self.user_multiplier = np.random.uniform(0.8, 1.2)
            elif self.workload_phase == 1:  # High load
                self.user_multiplier = np.random.uniform(1.5, 3.0)
            else:  # Low load
                self.user_multiplier = np.random.uniform(0.3, 0.7)
            
            self._phase_changed = True
            return True
        return False

    ################################################
    # reset
    ################################################
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.previous_algorithm = None
        self.workload_phase = 0
        self.phase_duration = 0
        self.user_multiplier = 1.0
        self._phase_changed = False

        # Random starting user count from actual data
        unique_users = sorted(self.df["users"].unique())
        self.base_users = np.random.choice(unique_users)

        # Get initial observation
        row = self.get_algorithm_performance(self.algorithms[0], self.base_users)
        obs = self._get_observation(row)
        
        return obs, {}

    ################################################
    # get observation with workload phase
    ################################################
    def _get_observation(self, row):
        """Create observation including workload phase with clipping"""
        # Apply dynamic user multiplier
        current_users = self.base_users * self.user_multiplier
        
        obs = np.array([
            self.normalize(current_users, "users"),
            self.normalize(row.security_bits, "security_bits"),
            self.normalize(row.cpu_mean_m, "cpu_mean_m"),
            self.normalize(row.memory_mean_mi, "memory_mean_mi"),
            self.normalize(row.latency_mean_ms, "latency_mean_ms"),
            self.normalize(row.throughput_mean_rps, "throughput_mean_rps"),
            self.workload_phase / 2.0  # Normalize phase (0, 0.5, 1.0)
        ], dtype=np.float32)
        
        # Clip to ensure within [0, 1] range
        return np.clip(obs, 0.0, 1.0)

    ################################################
    # STEP - Universal action handler for all algorithms
    ################################################
    def step(self, action):
        # ============================================
        # UNIVERSAL ACTION HANDLER
        # Works for: Q-learning, PPO, A2C, MAPPO
        # ============================================
        # Handle tuple/list/array (MAPPO)
        if isinstance(action, (tuple, list, np.ndarray)):
            action = int(action[0]) if len(action) > 0 else 0
        else:
            # Handle integer (Q-learning, PPO, A2C)
            action = int(action)
        
        # Ensure action is within valid range
        action = max(0, min(action, len(self.algorithms) - 1))
        
        # Update dynamic workload phase
        phase_changed = self.update_workload_phase()
        
        selected = self.algorithms[action]
        
        # Apply dynamic user multiplier
        current_users = self.base_users * self.user_multiplier
        
        # Find matching benchmark data
        tolerance = 0.2
        candidate = self.df[
            (self.df.users >= current_users * (1 - tolerance)) &
            (self.df.users <= current_users * (1 + tolerance)) &
            (self.df.key_size == selected)
        ]
        
        if len(candidate) == 0:
            # Find nearest users value
            nearest_idx = (self.df['users'] - current_users).abs().argsort()[:1]
            nearest_users = self.df.iloc[nearest_idx]['users'].values[0]
            candidate = self.df[
                (self.df.users == nearest_users) &
                (self.df.key_size == selected)
            ]
        
        if len(candidate) == 0:
            candidate = self.df.sample(1)
        
        row = candidate.iloc[0]

        ################################################
        # migration metrics (with phase influence)
        ################################################
        switch_time = 0
        service_interrupt = 0
        jwt_continuity = 100

        if self.previous_algorithm is not None:
            if selected != self.previous_algorithm:
                # Switch cost depends on workload phase
                if self.workload_phase == 1:  # High load = higher cost
                    switch_time = np.random.uniform(15, 45)
                    service_interrupt = np.random.uniform(5, 15)
                elif self.workload_phase == 2:  # Low load = lower cost
                    switch_time = np.random.uniform(2, 10)
                    service_interrupt = np.random.uniform(0.5, 3)
                else:  # Normal load
                    switch_time = np.random.uniform(5, 30)
                    service_interrupt = np.random.uniform(1, 10)
                
                jwt_continuity = max(85, 100 - service_interrupt)

        self.previous_algorithm = selected

        ################################################
        # reward calculation with switching incentives
        ################################################
        base_reward = (
            0.30 * self.normalize(row.security_bits, "security_bits") +
            0.20 * self.normalize(row.throughput_mean_rps, "throughput_mean_rps") -
            0.20 * self.normalize(row.latency_mean_ms, "latency_mean_ms") -
            0.15 * self.normalize(row.cpu_mean_m, "cpu_mean_m") -
            0.10 * self.normalize(row.memory_mean_mi, "memory_mean_mi")
        )
        
        # Switch penalties/bonuses
        switch_penalty = 0
        exploration_bonus = 0
        
        if self.previous_algorithm is not None:
            if selected == self.previous_algorithm:
                # Small penalty for staying (encourages exploration)
                if self.workload_phase == 1:
                    switch_penalty = -0.03
                else:
                    switch_penalty = -0.01
            else:
                # Bonus for trying new algorithms
                exploration_bonus = 0.02
        
        # Phase adaptation bonus
        phase_bonus = 0.05 if phase_changed else 0
        
        reward = (
            base_reward -
            0.03 * (switch_time / 30) -
            0.02 * (service_interrupt / 10) +
            0.05 * (jwt_continuity / 100) +
            switch_penalty +
            exploration_bonus +
            phase_bonus
        )

        ################################################
        # next state (with dynamic workload)
        ################################################
        self.idx = np.random.randint(len(self.df))
        row2 = self.df.iloc[self.idx]
        self.base_users = row2.users
        
        obs = self._get_observation(row2)

        ################################################
        # info dictionary
        ################################################
        info = {
            "latency": row.latency_mean_ms,
            "throughput": row.throughput_mean_rps,
            "cpu": row.cpu_mean_m,
            "memory": row.memory_mean_mi,
            "security_bits": row.security_bits,
            "switch_time": switch_time,
            "service_interrupt": service_interrupt,
            "jwt_continuity": jwt_continuity,
            "workload_phase": self.workload_phase,
            "user_multiplier": self.user_multiplier
        }

        ################################################
        # stop episode
        ################################################
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, info