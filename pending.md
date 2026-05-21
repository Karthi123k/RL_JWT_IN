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