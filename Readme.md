
# RL Integration Plan for RL_JWT_IN

## Project Understanding

The project already contains:

- Static benchmark evaluation for PQC algorithms
- JWT-based authentication microservices
- Kubernetes/K3d deployment
- Performance metrics collection
- PQC algorithms such as:
  - ML-DSA-44
  - ML-DSA-65
  - ML-DSA-87
  - Flacon variants
  - SPHINCS+ variants

Current implementation performs static evaluation only.

Current flow:

User Request
→ JWT Service
→ Fixed PQC Algorithm
→ Token Generation
→ Metrics Collection

---

## Goal

Integrate Reinforcement Learning into the existing RL_JWT_IN project so algorithm selection becomes adaptive rather than fixed.

Instead of:

"Use one predefined algorithm"

The system becomes:

"Select the best algorithm dynamically according to system conditions."

---

## Target Architecture

User Request
→ RL Agent
→ Observe State
→ Select PQC Algorithm
→ JWT Service
→ Generate Token
→ Collect Metrics
→ Reward Calculation
→ RL Policy Update

---

## RL Integration Components

### 1. RL Engine Module

Proposed structure:

RL_JWT_IN/

auth-service/
gateway/
k8s/
benchmarks/

rl_engine/
    environment.py
    agent.py
    reward.py
    train.py
    inference.py

results/

---

### 2. State Space

Static benchmark metrics become RL states.

Example state:

state=[
    security_level,
    cpu_usage,
    memory_usage,
    request_load,
    latency
]

Example:

state=[192,45,26,300,20]

Meaning:

- Security level = 192 bits
- CPU usage = 45%
- Memory usage = 26 MB
- Request load = 300
- Latency = 20 ms

---

### 3. Action Space

Actions represent algorithm selection.

actions={

0:"MLDSA44",
1:"MLDSA65",
2:"MLDSA87",
3:"SPHINCS192f",
4:"SPHINCS256f"

}

RL agent chooses one action.

---

## Environment Design

Environment responsibilities:

1. Observe current system metrics
2. Execute selected algorithm
3. Measure:

   - latency
   - CPU
   - memory
   - security

4. Compute reward
5. Return next state

---

## Reward Function

Multi-objective reward:

R=w1S−w2L−w3C−w4M

Where:

S = security score

L = latency

C = CPU usage

M = memory usage

Example:

reward=(0.4*security)
        -(0.3*latency)
        -(0.2*cpu)
        -(0.1*memory)

Objective:

Maximize:

- Security

Minimize:

- Latency
- CPU utilization
- Memory utilization

---

## Training Phase

Training loop:

Reset environment

Observe state

Select action

Execute algorithm

Calculate reward

Update Q-table/PPO policy

Repeat

---

## Deployment Flow

Gateway
→ RL Agent API
→ Auth Service
→ JWT Generation

The RL agent becomes an independent microservice.

---

## Reseamkdir rl_engine
cd rl_engine

touch environment.py
touch agent.py
touch reward.py
touch train.py
touch inference.pyrch Contribution

Static benchmarking:

- Identifies algorithm characteristics

RL integration:

- Learns optimal algorithm selection dynamically

Combined contribution:

"Static benchmarking provides algorithm performance knowledge, while reinforcement learning enables adaptive crypto-agile selection under changing workload conditions."

---

## Expected Outcome

The system should:

✓ Dynamically switch PQC algorithms

✓ Adapt to workload changes

✓ Reduce latency

✓ Reduce resource usage

✓ Maintain security requirements

✓ Enable crypto agility for JWT authentication systems

---

## Future Extension

- Replace synthetic environment with Prometheus live metrics
- Compare Q-learning and PPO
- Add MARL for distributed environments
- Enable online learning in Kubernetes clusters