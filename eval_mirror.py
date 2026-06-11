"""
eval_mirror.py — Mirror match evaluation (headless)
Runs N episodes of Model A (agents[0]) vs Model B (agents[1])
and prints win/draw/loss statistics.
"""

import os, math, argparse
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from training.env import HaxballCurriculumEnv

def run_mirror(model_path_a, model_path_b, n_episodes=200, max_steps=1350):
    print(f"Loading Model A: {model_path_a}")
    model_a = PPO.load(model_path_a, device="cpu")
    print(f"Loading Model B: {model_path_b}")
    model_b = PPO.load(model_path_b, device="cpu")

    env = HaxballCurriculumEnv(phase="A1")
    env.forced_opponent_type = "Trained"
    env.opponent_policy = model_b

    wins_a = 0
    wins_b = 0
    draws   = 0

    for ep in range(n_episodes):
        # Alternate sides: even episodes A=RED, odd episodes A=BLUE
        obs, _ = env.reset()
        if ep % 2 == 0:
            env.team_id      = 1
            env._flip        = 1.0
            env._attack_sign = 1
        else:
            env.team_id      = 2
            env._flip        = -1.0
            env._attack_sign = -1
        env._reset_positions()
        a = env.agents[0]
        env._prev_dist_to_ball = math.hypot(a.x - env.ball.x, a.y - env.ball.y)

        obs = env._get_obs()
        done = False
        step = 0

        while not done and step < max_steps:
            action, _ = model_a.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

        # Determine winner from scores
        # scores[0] = RED score, scores[1] = BLUE score
        red_score  = env.scores[0]
        blue_score = env.scores[1]
        agent_a_is_red = (ep % 2 == 0)

        score_a = red_score  if agent_a_is_red else blue_score
        score_b = blue_score if agent_a_is_red else red_score

        if score_a > score_b:
            wins_a += 1
            result = "A wins"
        elif score_b > score_a:
            wins_b += 1
            result = "B wins"
        else:
            draws += 1
            result = "Draw"

        if (ep + 1) % 20 == 0:
            print(f"  [{ep+1:3d}/{n_episodes}] A:{wins_a}  B:{wins_b}  D:{draws}  ({result})")

    total = wins_a + wins_b + draws
    print("\n══════════════════════════════════")
    print(f"  Kết quả sau {n_episodes} trận:")
    print(f"  Model A (agents[0]): {wins_a}/{total} = {wins_a/total*100:.1f}%")
    print(f"  Model B (agents[1]): {wins_b}/{total} = {wins_b/total*100:.1f}%")
    print(f"  Hòa:                 {draws}/{total} = {draws/total*100:.1f}%")
    print("══════════════════════════════════")

    env.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-a", default="models/experiment/a1_checkpoints/snapshot_1000000.zip")
    p.add_argument("--model-b", default="models/experiment/a1_checkpoints/snapshot_1000000.zip")
    p.add_argument("--episodes", default=200, type=int)
    args = p.parse_args()

    run_mirror(args.model_a, args.model_b, n_episodes=args.episodes)
