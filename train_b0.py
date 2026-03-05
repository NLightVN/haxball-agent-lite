"""
train_b0.py — B0 Defend Bot Training
======================================
B0 learns to defend goal_B0 against a frozen A0 attacker.

Usage:
    python train_b0.py                                         # A0 = heuristic follower
    python train_b0.py --a0-model models/a0_best.zip          # A0 = trained PPO
    python train_b0.py --a0-model models/a0_best.zip --steps 5000000
    python train_b0.py --resume models/b0_checkpoints/b0_100000_steps.zip
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList,
)

from training.env_b0 import HaxballB0Env

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_ENVS          = 8
TOTAL_STEPS     = 5_000_000
CHECKPOINT_FREQ = 50_000
TARGET_REWARD   = 2.0        # early-stop when ep_rew_mean >= this

PPO_PARAMS = dict(
    n_steps       = 512,
    batch_size    = 256,
    n_epochs      = 8,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    ent_coef      = 0.02,
    learning_rate = 3e-4,
    clip_range    = 0.2,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
)


class B0MonitorCallback(BaseCallback):
    """Logs progress, saves best model, and stops early when target is reached."""

    def __init__(self, target_reward: float, model_dir: str, verbose=1):
        super().__init__(verbose)
        self.target_reward = target_reward
        self.model_dir     = model_dir
        self.best_reward   = -np.inf
        self.best_path     = os.path.join(model_dir, "b0_best")

    def _on_step(self) -> bool:
        rew = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
        if rew is None:
            return True

        if rew > self.best_reward:
            self.best_reward = rew
            self.model.save(self.best_path)
            if self.verbose:
                print(f"[B0] ✨ New best reward: {rew:.4f}  → {self.best_path}.zip")

        if rew >= self.target_reward:
            print(f"\n[B0] 🎯 Target reached! ep_rew_mean={rew:.4f} >= {self.target_reward}")
            return False

        return True


def parse_args():
    p = argparse.ArgumentParser(description="Train B0 defend bot")
    p.add_argument("--a0-model", default=None,
                   help="Path to trained A0 .zip (omit for heuristic attacker)")
    p.add_argument("--steps",    default=TOTAL_STEPS, type=int,
                   help="Max training timesteps")
    p.add_argument("--resume",   default=None,
                   help="Resume from a B0 checkpoint .zip")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs("models/b0_checkpoints", exist_ok=True)

    def make_env():
        return HaxballB0Env(a0_model_path=args.a0_model)

    vec_env = DummyVecEnv([make_env for _ in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    if args.resume:
        print(f"[B0] Resuming from {args.resume}")
        model = PPO.load(args.resume, env=vec_env, device="auto")
        remaining = max(args.steps - model.num_timesteps, 0)
        reset_ts  = False
    else:
        print(f"[B0] Starting fresh B0 training")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose       = 1,
            tensorboard_log = "./tensorboard/b0/",
            **PPO_PARAMS,
        )
        remaining = args.steps
        reset_ts  = True

    a0_label = args.a0_model if args.a0_model else "heuristic follower"
    print(f"[B0] A0 opponent  : {a0_label}")
    print(f"[B0] Envs         : {N_ENVS}")
    print(f"[B0] Steps        : {remaining:,}")
    print(f"[B0] Checkpoints  : models/b0_checkpoints/  |  Best: models/b0_best.zip")
    print(f"[B0] TensorBoard  : tensorboard/b0/\n")

    checkpoint_cb = CheckpointCallback(
        save_freq   = max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path   = "models/b0_checkpoints/",
        name_prefix = "b0",
        verbose     = 1,
    )
    monitor_cb = B0MonitorCallback(
        target_reward = TARGET_REWARD,
        model_dir     = "models/",
        verbose       = 1,
    )

    model.learn(
        total_timesteps     = remaining,
        callback            = CallbackList([checkpoint_cb, monitor_cb]),
        reset_num_timesteps = reset_ts,
        progress_bar        = True,
    )

    final_path = "models/b0_final"
    model.save(final_path)
    print(f"\n[B0] Training complete → {final_path}.zip")
