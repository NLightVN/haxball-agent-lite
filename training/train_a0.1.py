"""
train_a0.1.py — Early Curriculum Training (Phase A0.1).

1 step = 3 ticks.
Episode time = 3 minutes.
Custom curriculum on goal width over 1M steps.
"""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.env import HaxballCurriculumEnv

N_ENVS          = 8
CHECKPOINT_FREQ = 250_000

PPO_PARAMS = dict(
    n_steps      = 512,
    batch_size   = 256,
    n_epochs     = 8,
    gamma        = 0.99,
    gae_lambda   = 0.98,
    ent_coef     = 0.02,
    learning_rate= 2e-4,
    clip_range   = 0.2,
    vf_coef      = 0.5,
    max_grad_norm= 0.5,
)

class A01Callback(BaseCallback):
    def __init__(self, model_dir, envs, save_freq, verbose=1):
        super().__init__(verbose)
        self.model_dir = model_dir
        self.envs = envs
        self.save_freq = max(save_freq // N_ENVS, 1)
        self.best_reward = -np.inf
        self.best_path = os.path.join(model_dir, "a0.1_best")
        
    def _on_step(self) -> bool:
        # Sync timesteps to the env
        for env in self.envs.envs:
            env.total_timesteps_elapsed = self.num_timesteps
            
        if self.n_calls % self.save_freq == 0:
            snapshot_name = f"a0.1_snapshot_{self.num_timesteps}"
            path = os.path.join(self.model_dir, "a0.1_checkpoints", f"{snapshot_name}.zip")
            self.model.save(path)
            if self.verbose:
                print(f"[A0.1] Saved snapshot to {path}")

        rew = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
        if rew is not None and rew > self.best_reward:
            self.best_reward = rew
            self.model.save(self.best_path)

        return True

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",    default=6_000_000,  type=int)
    p.add_argument("--resume",   default=None,        help="Resume from a previous checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs("models/a0.1_checkpoints", exist_ok=True)

    def make_env():
        return HaxballCurriculumEnv(phase='A0.1', n_agents=1)

    vec_env = DummyVecEnv([make_env for _ in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    if args.resume:
        print(f"[A0.1] Resuming from {args.resume}")
        model = PPO.load(args.resume, env=vec_env, custom_objects=PPO_PARAMS)
        model.set_env(vec_env)
    else:
        print(f"[A0.1] Starting completely fresh")
        model = PPO("MlpPolicy", vec_env, verbose=1, device="auto", **PPO_PARAMS)

    callback = A01Callback(
        model_dir="models/",
        envs=vec_env,
        save_freq=CHECKPOINT_FREQ,
        verbose=1
    )

    is_resume = bool(args.resume)
    remaining_steps = max(args.steps - model.num_timesteps, 0) if is_resume else args.steps

    print(f"[A0.1] Starting — {N_ENVS} envs, {remaining_steps:,} steps to go")

    model.learn(
        total_timesteps      = remaining_steps,
        callback             = callback,
        reset_num_timesteps  = not is_resume,
        log_interval         = 250_000 // (PPO_PARAMS["n_steps"] * N_ENVS),
        progress_bar         = True,
    )

    model.save("models/a0.1_final")
    print("\n[A0.1] Done -> models/a0.1_final.zip")
