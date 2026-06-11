"""
eval.py — Evaluate a trained A0 (or A1) model.

Usage:
    python eval.py --model models/experiment/a0_best.zip
    python eval.py --model models/experiment/a0_best.zip --render --delay 0.05
    python eval.py --model models/experiment/a0_best.zip --render --goal-scale 0.5
"""

import argparse
import os
import time
import numpy as np
from stable_baselines3 import PPO
from training.env import HaxballCurriculumEnv


# ── ASCII Renderer ─────────────────────────────────────────────────────────────
FIELD_W = 80   # columns
FIELD_H = 22   # rows

def render_frame(env, step, reward, ep_reward, result_str=""):
    """Draw an ASCII top-down view of the field."""
    HW, HH = env.HW, env.HH
    gy = env.goal_y
    gcy = env.goal_center_y

    grid = [[' '] * FIELD_W for _ in range(FIELD_H)]

    def to_col(x): return int((x + HW) / (2 * HW) * (FIELD_W - 1))
    def to_row(y): return int((y + HH) / (2 * HH) * (FIELD_H - 1))

    # Field border
    for c in range(FIELD_W):
        grid[0][c] = '─'
        grid[FIELD_H - 1][c] = '─'
    for r in range(FIELD_H):
        grid[r][0] = '│'
        grid[r][FIELD_W - 1] = '│'

    # Goal posts (left & right)
    for side_x in [-HW, HW]:
        top_y = gcy - gy
        bot_y = gcy + gy
        for y in np.linspace(top_y, bot_y, 6):
            r, c = to_row(y), to_col(side_x)
            if 0 <= r < FIELD_H and 0 <= c < FIELD_W:
                grid[r][c] = 'G'

    # Center line
    cx = to_col(0)
    for r in range(1, FIELD_H - 1):
        if grid[r][cx] == ' ':
            grid[r][cx] = '┆'

    # Ball
    br, bc = to_row(env.ball.y), to_col(env.ball.x)
    if 0 <= br < FIELD_H and 0 <= bc < FIELD_W:
        grid[br][bc] = 'O'

    # Agents
    icons = ['P', 'E']  # P=our agent, E=enemy
    for i, ag in enumerate(env.agents):
        ar, ac = to_row(ag.y), to_col(ag.x)
        if 0 <= ar < FIELD_H and 0 <= ac < FIELD_W:
            grid[ar][ac] = icons[i] if i < len(icons) else 'A'

    # Build string
    os.system('cls' if os.name == 'nt' else 'clear')
    lines = [''.join(row) for row in grid]
    attack = "→ RIGHT" if env._attack_sign == 1 else "← LEFT"
    team   = "RED" if env.team_id == 1 else "BLUE"
    header = (f"  Step {step:3d}/{env.max_steps}  |  Rew {ep_reward:+.3f}"
              f"  |  {team} ({attack})  |  goal_y={gy:.0f}  {result_str}")
    print(f"{'─'*FIELD_W}\n{header}\n{'─'*FIELD_W}")
    for l in lines:
        print(l)
    print(f"{'─'*FIELD_W}  O=ball  P=agent  E=enemy  G=goal  ┆=center")


# ── Main ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    required=True,  help="Path to .zip model file")
    p.add_argument("--phase",    default="A0",   help="A0 or A1 (default: A0)")
    p.add_argument("--episodes", default=100,    type=int,   help="Number of episodes")
    p.add_argument("--render",      action="store_true",        help="Show ASCII field")
    p.add_argument("--delay",        default=0.05,   type=float, help="Seconds per frame (default 0.05)")
    p.add_argument("--goal-scale",   default=1.0,    type=float, help="Multiply goal_y after reset (e.g. 0.5 = half-width goal)")
    p.add_argument("--deterministic", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model, device="cpu")
    env   = HaxballCurriculumEnv(phase=args.phase)

    rewards, ep_lengths = [], []
    goals_scored = timeouts = own_goals = 0
    goal_scale = args.goal_scale

    if goal_scale != 1.0:
        print(f"[eval] goal-scale={goal_scale}  (goal_y will be ×{goal_scale} after each reset)")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        # Override goal size for eval difficulty
        if goal_scale != 1.0:
            env.goal_y *= goal_scale
            # Rebuild obs with corrected goal_y
            obs = env._get_obs()
        ep_reward = 0.0
        ep_len    = 0
        result    = "timeout"
        done      = False

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_len    += 1
            done = terminated or truncated

            if terminated:
                result = ("goal"     if args.phase == "A0" and reward > 0
                          else "own_goal" if args.phase == "A0"
                          else "win"  if env.scores[env.team_id - 1] >= 3
                          else "loss")
            elif truncated:
                result = "timeout"

            if args.render:
                render_frame(env, ep_len, reward, ep_reward,
                             f"← {result.upper()}" if done else "")
                time.sleep(args.delay)

        rewards.append(ep_reward)
        ep_lengths.append(ep_len)
        if result == "goal":     goals_scored += 1
        elif result == "timeout": timeouts    += 1
        elif result == "own_goal": own_goals  += 1

        if not args.render:
            bar = "✅" if result == "goal" else ("⏱" if result == "timeout" else "❌")
            print(f"  Ep {ep+1:4d} {bar} | {result:8s} | rew {ep_reward:+.3f} | steps {ep_len}")

    # ── Summary ────────────────────────────────────────────────────────────────
    n = args.episodes
    print(f"\n{'='*50}")
    print(f"  Model : {args.model}")
    print(f"  Phase : {args.phase}  |  Episodes: {n}")
    print(f"{'='*50}")
    if args.phase == "A0":
        print(f"  ✅ Goals   : {goals_scored}/{n}  ({goals_scored/n*100:.1f}%)")
        print(f"  ⏱  Timeout : {timeouts}/{n}")
        print(f"  ❌ OwnGoal : {own_goals}/{n}")
    else:
        print(f"  Wins   : {goals_scored}/{n}  ({goals_scored/n*100:.1f}%)")
        print(f"  Losses : {timeouts}/{n}")
    print(f"  Mean reward : {np.mean(rewards):+.3f}  ± {np.std(rewards):.3f}")
    print(f"  Mean ep len : {np.mean(ep_lengths):.1f} / {env.max_steps}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
