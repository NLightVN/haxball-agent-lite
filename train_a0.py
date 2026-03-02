"""
Train A0 only — 1 agent, no opponent, score into right goal.
- Checkpoints every 50,000 steps → models/a0_checkpoints/
- Auto-stop when rollout/ep_rew_mean >= 3.0 (consistently scoring)
- Saves best model to models/a0_best.zip
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from training.env import HaxballCurriculumEnv

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_ENVS          = 8          # parallel envs
TOTAL_STEPS     = 5_000_000  # max training steps
CHECKPOINT_FREQ = 50_000     # save every N steps (across all envs)
TARGET_REWARD   = 3.0        # stop early when ep_rew_mean >= this
# ── PPO config ─────────────────────────────────────────────────────────────────
PPO_PARAMS = dict(
    n_steps      = 512,    # rollout buffer per env
    batch_size   = 256,
    n_epochs     = 8,
    gamma        = 0.99,
    gae_lambda   = 0.95,
    ent_coef     = 0.02,   # encourage exploration
    learning_rate= 3e-4,
    clip_range   = 0.2,
    vf_coef      = 0.5,
    max_grad_norm= 0.5,
)


class A0MonitorCallback(BaseCallback):
    """Logs progress and stops early when target reward is reached."""

    def __init__(self, target_reward: float, model_dir: str, verbose=1):
        super().__init__(verbose)
        self.target_reward  = target_reward
        self.model_dir      = model_dir
        self.best_reward    = -np.inf
        self.best_path      = os.path.join(model_dir, "a0_best")

    def _on_step(self) -> bool:
        rew = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
        if rew is None:
            return True

        # Save best model whenever reward improves
        if rew > self.best_reward:
            self.best_reward = rew
            self.model.save(self.best_path)
            if self.verbose:
                print(f"[A0] ✨ New best reward: {rew:.3f}  → saved to {self.best_path}.zip")

        # Early stop
        if rew >= self.target_reward:
            print(f"\n[A0] 🎯 Target reached! ep_rew_mean={rew:.3f} >= {self.target_reward}")
            print(f"[A0] Best model at: {self.best_path}.zip")
            return False  # stops training

        return True


if __name__ == "__main__":
    os.makedirs("models/a0_checkpoints", exist_ok=True)

    # Create N_ENVS parallel A0 environments
    vec_env = DummyVecEnv([lambda: HaxballCurriculumEnv(phase='A0') for _ in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./tensorboard/a0/",
        **PPO_PARAMS,
    )

    # Checkpoint every CHECKPOINT_FREQ steps (adjusted for N_ENVS)
    checkpoint_cb = CheckpointCallback(
        save_freq = max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path = "models/a0_checkpoints/",
        name_prefix = "a0",
        verbose = 1,
    )

    monitor_cb = A0MonitorCallback(
        target_reward = TARGET_REWARD,
        model_dir     = "models/",
        verbose       = 1,
    )

    print(f"[A0] Starting training — {N_ENVS} envs, up to {TOTAL_STEPS:,} steps")
    print(f"[A0] Checkpoints: models/a0_checkpoints/  |  Best: models/a0_best.zip")
    print(f"[A0] Tensorboard: tensorboard/a0/\n")

    model.learn(
        total_timesteps = TOTAL_STEPS,
        callback        = CallbackList([checkpoint_cb, monitor_cb]),
        progress_bar    = True,
    )

    # Final save (if not early-stopped)
    final_path = "models/a0_final"
    model.save(final_path)
    print(f"\n[A0] Training complete. Final model: {final_path}.zip")
