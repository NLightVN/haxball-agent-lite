"""
train_a0.1.py — Early Curriculum Training (Phase A0.1).

1 step = 3 ticks.
Episode time = 1 minute.
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
from utils.model_logger import get_model_logger

log = get_model_logger("a0.1")

N_ENVS          = 8
CHECKPOINT_FREQ = 250_000

PPO_PARAMS = dict(
    n_steps      = 2048,
    batch_size   = 512,
    n_epochs     = 10,
    gamma        = 0.99,
    gae_lambda   = 0.95,
    ent_coef     = 0.02,
    learning_rate= 3e-4,
    clip_range   = 0.2,
    vf_coef      = 0.5,
    max_grad_norm= 0.5,
    policy_kwargs= dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
)

class A01Callback(BaseCallback):
    def __init__(self, model_dir, envs, save_freq, verbose=1):
        super().__init__(verbose)
        self.model_dir = model_dir
        self.envs = envs
        self.save_freq = max(save_freq // N_ENVS, 1)
        self.best_reward = -np.inf
        self.best_path = os.path.join(model_dir, "a0.1_best")
        self._a0_1_ball_goal_record_reward_total = 0.0
        self._a0_1_episode_count = 0
        self._last_logged_iteration = -1

    # Keys to pull from SB3's internal logger and write to file
    _METRIC_KEYS = [
        ("rollout/ep_len_mean",       "ep_len_mean"),
        ("rollout/ep_rew_mean",        "ep_rew_mean"),
        ("time/fps",                   "fps"),
        ("time/iterations",            "iterations"),
        ("time/time_elapsed",          "time_elapsed"),
        ("time/total_timesteps",       "total_timesteps"),
        ("train/approx_kl",            "approx_kl"),
        ("train/clip_fraction",        "clip_fraction"),
        ("train/clip_range",           "clip_range"),
        ("train/entropy_loss",         "entropy_loss"),
        ("train/explained_variance",   "explained_variance"),
        ("train/learning_rate",        "learning_rate"),
        ("train/loss",                 "loss"),
        ("train/n_updates",            "n_updates"),
        ("train/policy_gradient_loss", "policy_gradient_loss"),
        ("train/value_loss",           "value_loss"),
    ]

    def _log_metrics_to_file(self):
        """Dump current SB3 metrics to the file logger in a readable format."""
        kv = self.logger.name_to_value
        lines = ["─" * 52]
        for sb3_key, label in self._METRIC_KEYS:
            val = kv.get(sb3_key)
            if val is not None:
                lines.append(f"  {label:<28} {val}")
        lines.append("─" * 52)
        log.info("\n" + "\n".join(lines))

    def _on_step(self) -> bool:
        # Sync timesteps to the env
        for env in self.envs.envs:
            env.total_timesteps_elapsed = self.num_timesteps

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, done in enumerate(dones):
            if done and i < len(self.envs.envs):
                env = self.envs.envs[i]
                self._a0_1_ball_goal_record_reward_total += float(getattr(env, "_a0_1_record_ball_to_goal_reward_total", 0.0))
                self._a0_1_episode_count += 1

        if self._a0_1_episode_count > 0:
            self.logger.record(
                "a0.1/ball_goal_record_reward_mean",
                self._a0_1_ball_goal_record_reward_total / self._a0_1_episode_count,
            )
            if infos:
                last_info = infos[-1]
                if isinstance(last_info, dict):
                    self.logger.record(
                        "a0.1/best_ball_dist_to_goal",
                        float(last_info.get("a0_1/best_ball_dist_to_goal", 0.0)),
                    )
                    self.logger.record(
                        "a0.1/cur_ball_dist_to_goal",
                        float(last_info.get("a0_1/cur_ball_dist_to_goal", 0.0)),
                    )
                    self.logger.record(
                        "a0.1/steps_since_ball_record",
                        float(last_info.get("a0_1/steps_since_ball_record", 0.0)),
                    )

        # ── Write metrics to file log whenever SB3 logs a new iteration ───────
        cur_iter = int(self.logger.name_to_value.get("time/iterations", -1))
        if cur_iter > self._last_logged_iteration and cur_iter >= 0:
            self._last_logged_iteration = cur_iter
            self._log_metrics_to_file()

        if self.n_calls % self.save_freq == 0:
            snapshot_name = f"a0.1_snapshot_{self.num_timesteps}"
            path = os.path.join(self.model_dir, "a0.1_checkpoints", f"{snapshot_name}.zip")
            self.model.save(path)
            if self.verbose:
                log.info(f"[A0.1] Saved snapshot to {path}")

        rew = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
        if rew is not None and rew > self.best_reward:
            self.best_reward = rew
            self.model.save(self.best_path)

        return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",    default=2_000_000,  type=int)
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
        log.info(f"[A0.1] Resuming from {args.resume}")
        custom_objects = PPO_PARAMS.copy()
        if "policy_kwargs" in custom_objects:
            del custom_objects["policy_kwargs"]
        model = PPO.load(args.resume, env=vec_env, custom_objects=custom_objects)
        model.set_env(vec_env)
    else:
        log.info(f"[A0.1] Starting completely fresh")
        model = PPO("MlpPolicy", vec_env, verbose=1, device="auto", **PPO_PARAMS)

    callback = A01Callback(
        model_dir="models/",
        envs=vec_env,
        save_freq=CHECKPOINT_FREQ,
        verbose=1
    )

    is_resume = bool(args.resume)
    remaining_steps = max(args.steps - model.num_timesteps, 0) if is_resume else args.steps

    log.info(f"[A0.1] Starting — {N_ENVS} envs, {remaining_steps:,} steps to go")

    model.learn(
        total_timesteps      = remaining_steps,
        callback             = callback,
        reset_num_timesteps  = not is_resume,
        log_interval         = 250_000 // (PPO_PARAMS["n_steps"] * N_ENVS),
        progress_bar         = True,
    )

    model.save("models/a0.1_final")
    log.info("[A0.1] Done -> models/a0.1_final.zip")
