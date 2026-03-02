"""
train_a1.py — A1 training: dual-episode curriculum.

Episode types (alternating 50/50 each reset):
  • PRECISION : small goal (0.3–0.6×), no opponent → aim accuracy
  • OPPONENT  : goal 1.4×→0.6× over 2M steps, opponent = Follower/Defender or A0

Usage:
    python train_a1.py --a0-model models/a0_best.zip
    python train_a1.py --a0-model models/a0_best.zip --steps 5000000
"""

import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from training.env import HaxballCurriculumEnv

# ── Config ─────────────────────────────────────────────────────────────────────
N_ENVS          = 8
CHECKPOINT_FREQ = 50_000
TARGET_REWARD   = 3.5        # early-stop threshold (harder than A0 due to opponent)

PPO_PARAMS = dict(
    n_steps      = 512,
    batch_size   = 256,
    n_epochs     = 8,
    gamma        = 0.99,
    gae_lambda   = 0.95,
    ent_coef     = 0.02,
    learning_rate= 2e-4,     # slightly lower than A0 for fine-tuning stability
    clip_range   = 0.2,
    vf_coef      = 0.5,
    max_grad_norm= 0.5,
)


class A1MonitorCallback(BaseCallback):
    def __init__(self, target_reward, model_dir, envs, verbose=1):
        super().__init__(verbose)
        self.target_reward = target_reward
        self.model_dir     = model_dir
        self.envs          = envs
        self.best_reward   = -np.inf
        self.best_path     = os.path.join(model_dir, "a1_best")

    def _on_step(self) -> bool:
        # Sync timesteps so env curriculum sees current progress
        for env in self.envs.envs:
            env.total_timesteps_elapsed = self.num_timesteps

        rew = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
        if rew is None:
            return True

        if rew > self.best_reward:
            self.best_reward = rew
            self.model.save(self.best_path)
            if self.verbose:
                print(f"[A1] New best: {rew:.3f} -> {self.best_path}.zip")

        if rew >= self.target_reward:
            print(f"\n[A1] Target reached! ep_rew_mean={rew:.3f}")
            return False

        return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--a0-model", required=True,       help="Path to trained A0 model .zip")
    p.add_argument("--steps",    default=2_000_000,   type=int)
    p.add_argument("--resume",   default=None,        help="Resume from a previous A1 checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs("models/a1_checkpoints", exist_ok=True)

    def make_env():
        env = HaxballCurriculumEnv(phase='A1')
        env.a0_model_path = args.a0_model   # inject A0 model path for opponent sampling
        return env

    vec_env = DummyVecEnv([make_env for _ in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    if args.resume:
        print(f"[A1] Resuming from {args.resume}")
        model = PPO.load(args.resume, env=vec_env)
    else:
        # Start from A0 weights → much faster convergence
        print(f"[A1] Loading A0 weights from {args.a0_model}")
        model = PPO.load(args.a0_model, env=vec_env, device="auto")
        # Override PPO hyperparams for A1
        model.learning_rate  = PPO_PARAMS["learning_rate"]
        model.ent_coef       = PPO_PARAMS["ent_coef"]
        model.n_steps        = PPO_PARAMS["n_steps"]
        model.batch_size     = PPO_PARAMS["batch_size"]

    checkpoint_cb = CheckpointCallback(
        save_freq   = max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path   = "models/a1_checkpoints/",
        name_prefix = "a1",
        verbose     = 1,
    )
    monitor_cb = A1MonitorCallback(
        target_reward = TARGET_REWARD,
        model_dir     = "models/",
        envs          = vec_env,
        verbose       = 1,
    )

    is_resume = bool(args.resume)
    if is_resume:
        remaining_steps = max(args.steps - model.num_timesteps, 0)
        print(f"[A1] Resuming from step {model.num_timesteps:,} — {remaining_steps:,} steps remaining")
    else:
        remaining_steps = args.steps

    print(f"[A1] Starting — {N_ENVS} envs, {remaining_steps:,} steps to go")
    print(f"[A1] Dual-episode: 80% opponent | 20% precision")
    print(f"[A1] Goal curriculum: 1.4x -> 0.6x over 2M steps\n")

    model.learn(
        total_timesteps      = remaining_steps,
        callback             = CallbackList([checkpoint_cb, monitor_cb]),
        reset_num_timesteps  = not is_resume,   # False when resuming → keeps step counter
        progress_bar         = True,
    )

    model.save("models/a1_final")
    print("\n[A1] Done → models/a1_final.zip")
