Looking at your A2C results CSV, I can see the issue clearly. You're logging per-step rewards as separate "episodes" (rows 1-10048), but your earlier results table showed cumulative episode rewards (1,634 for A2C).
The Problem: Your Logger is Wrong

Your CSV shows:

    episode column = step number (1 to 10,048 steps)

    reward column = per-step reward (0.02 to 0.46 range)

    But you reported A2C cumulative reward = 1,634

This means:

    If average step reward ~0.2 × 50 steps/episode = ~10 per episode

    Your reported 1,634 would require ~163 episodes of training

    Your CSV has 10,048 rows = 10,048 steps = ~200 episodes (if 50 steps/episode)






    What Your Results Actually Show
Algorithm	Reward	What it actually represents
Q, PPO, A2C	1,600-3,200	Trained policies (correct)
MAPPO	367,210	Random policy (incorrect)



/home/karthik/Karthik/RL_JWT_IN/
│
├── 📁 k8s/                          # Kubernetes configurations
│   ├── auth.yaml
│   ├── user.yaml
│   ├── gateway.yaml
│   ├── namespace.yaml
│   └── rl-deployment.yaml           # NEW: RL model deployments
│
├── 📁 locust/                       # Load testing
│   ├── locust_reuse.py              # Your locust file
│   ├── run_tests.sh
│   └── logs/
│       └── reuse/
│           └── overall_benchmark.csv  # Static benchmark data
│
├── 📁 rl/                           # RL training code (EXISTING)
│   ├── pqc_env.py
│   ├── mara_jwt_env.py
│   ├── train_qlearning.py
│   ├── train_a2c.py
│   ├── train_ppo.py
│   ├── train_mara_jwt.py
│   ├── evaluate_all.py
│   ├── extract.py
│   ├── 📁 models/                   # Trained models (EXISTING)
│   │   ├── qlearning_final.pkl
│   │   ├── a2c_final.zip
│   │   ├── ppo_final.zip
│   │   └── 📁 mara_jwt/
│   │       ├── security_agent_policy.pt
│   │       ├── performance_agent_policy.pt
│   │       ├── resource_agent_policy.pt
│   │       ├── traffic_agent_policy.pt
│   │       └── coordinator_policy.pt
│   └── 📁 results/                  # Training results (EXISTING)
│       ├── qlearning_results.csv
│       ├── a2c_results.csv
│       ├── ppo_results.csv
│       └── mara_jwt_results.csv
│
├── 📁 rl-deployment/                # NEW: Deployment code for RL models
│   ├── Dockerfile                   # Docker image for RL service
│   ├── requirements.txt             # Python dependencies
│   ├── rl_decision_service.py       # Main service script
│   ├── deploy_models.sh             # Deploy all models to K8s
│   ├── test_models.sh               # Test models with Locust
│   ├── compare_results.py           # Compare RL vs static benchmark
│   └── 📁 results/                  # Deployment test results
│       ├── rl_test_results/
│       └── comparison/
│
├── 📁 certs/                        # PQC certificates (EXISTING)
│   ├── falcon512/
│   ├── falcon1024/
│   ├── mldsa44/
│   ├── mldsa65/
│   ├── mldsa87/
│   ├── sphincs128f/
│   ├── sphincs128s/
│   ├── sphincs192f/
│   ├── sphincs192s/
│   ├── sphincs256f/
│   └── sphincs256s/
│
├── auth-service/                    # Auth service (EXISTING)
├── user-service/                    # User service (EXISTING)
├── nginx/                           # Nginx config (EXISTING)
├── monitoring/                      # Prometheus config (EXISTING)
│
├── docker-compose.yml               # Docker compose (EXISTING)
└── requirements.txt                 # Python dependencies (EXISTING)




Runtime Adaptability
=
successful transitions
/
total transitions

JWT Continuity
=
verified tokens
/
generated tokens

Switching Time
=
algorithm_selection_end
-
algorithm_selection_start

Service Interruption
=
downtime during switch

Security Satisfaction
=
selected_security_bits
/
required_security_bits