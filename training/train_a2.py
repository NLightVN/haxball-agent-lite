"""
train_a2.py — 2v2 Self-Play Training.

- 20% matches are 1v1 (opponent sampled from full A1 pool).
- 80% matches are 2v1 combinatorial sweep (Teammate, Opp1).
- PoolManager prioritizes pool_phoi_hop over pool_ca_nhan.
"""

import argparse
import glob
import os
import sys
import itertools
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.env import HaxballCurriculumEnv
from utils.model_logger import get_model_logger

log = get_model_logger("a2_combinatorial")

# ── Config ─────────────────────────────────────────────────────────────────────
N_ENVS          = 8
TARGET_REWARD   = 15.0

PPO_PARAMS = dict(
    n_steps      = 2048,
    batch_size   = 512,
    n_epochs     = 10,
    gamma        = 0.99,
    gae_lambda   = 0.95,
    ent_coef     = 0.02,
    learning_rate= 2e-4,
    clip_range   = 0.2,
    vf_coef      = 0.5,
    max_grad_norm= 0.5,
    policy_kwargs= dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
)

# ── Pool Manager ───────────────────────────────────────────────────────────────
class PoolManager:
    def __init__(self):
        self.a1_full_pool = []
        self.pool_ca_nhan_teammate = []
        self.pool_ca_nhan_opponent = []
        self.pool_phoi_hop = []
        
        self.loaded_policies = {}
        
    def get_policy(self, path):
        if path not in self.loaded_policies:
            log.info(f"[PoolManager] Loading {path}...")
            self.loaded_policies[path] = PPO.load(path, device='cpu')
        return self.loaded_policies[path]

    def get_2v1_match(self):
        # Chọn đồng đội
        if len(self.pool_phoi_hop) > 0 and np.random.rand() < 0.90:
            teammate = np.random.choice(self.pool_phoi_hop)
        else:
            teammate = np.random.choice(self.pool_ca_nhan_teammate)
            
        # Chọn đối thủ (luôn từ pool_ca_nhan_opponent)
        opponent = np.random.choice(self.pool_ca_nhan_opponent)
        
        return (teammate, opponent)

    def get_1v1_match(self):
        if not self.a1_full_pool:
            return None
        return np.random.choice(self.a1_full_pool)

# ── Self-Play Environment Wrapper ─────────────────────────────────────────────
class SelfPlayEnvA2(HaxballCurriculumEnv):
    def __init__(self, pool_manager, **kwargs):
        super().__init__(**kwargs)
        self.pool_manager = pool_manager
        
    def _reset_positions(self):
        super()._reset_positions()
        
        if self.episode_type == 'opponent':
            if getattr(self, 'is_1v1', False):
                opp_path = self.pool_manager.get_1v1_match()
                if opp_path:
                    self.opponent_policies = [self.pool_manager.get_policy(opp_path)]
                self.teammate_policy = None
            else:
                tm_path, opp1_path = self.pool_manager.get_2v1_match()
                self.teammate_policy = self.pool_manager.get_policy(tm_path)
                self.opponent_policies = [
                    self.pool_manager.get_policy(opp1_path)
                ]
                
            self.opponent_type = 'Trained'

# ── Combined Callback ──────────────────────────────────────────────────────────
class SelfPlayCallback(BaseCallback):
    def __init__(self, pool_manager, target_reward, model_dir, envs, verbose=1):
        super().__init__(verbose)
        self.pool_manager = pool_manager
        self.target_reward = target_reward
        self.model_dir = model_dir
        self.envs = envs
        self.next_snapshot_at = 500_000
        self.best_reward = -np.inf
        self.best_path = os.path.join(model_dir, "a2_best")
        self.last_log_step = 0
        
        # Check existing A2 snapshots to calculate next_snapshot_at
        ckpt_dir = os.path.join(self.model_dir, "a2_checkpoints")
        if os.path.exists(ckpt_dir):
            existing = [f for f in os.listdir(ckpt_dir) if f.startswith("snapshot_") and f.endswith(".zip")]
            if existing:
                N = len(existing)
                accumulated = sum((k + 1) * 500_000 for k in range(N))
                next_interval = (N + 1) * 500_000
                self.next_snapshot_at = accumulated + next_interval
        
    def _on_training_start(self) -> None:
        log.info(f"[SelfPlay A2] Next snapshot at step {self.next_snapshot_at:,}")
        
    def _on_step(self) -> bool:
        for env in self.envs.envs:
            env.total_timesteps_elapsed = self.num_timesteps
            
        if self.num_timesteps - self.last_log_step >= 250_000:
            log.info(f"\n--- [Step {self.num_timesteps:,}] A2 Pool Stats ---")
            log.info(f"Pool Phối hợp: {len(self.pool_manager.pool_phoi_hop)} models")
            log.info(f"Pool Cá nhân (Đồng đội): {len(self.pool_manager.pool_ca_nhan_teammate)} models")
            log.info(f"Pool Cá nhân (Đối thủ): {len(self.pool_manager.pool_ca_nhan_opponent)} models")
            self.last_log_step = self.num_timesteps
            
        while self.next_snapshot_at <= self.num_timesteps:
            snapshot_name = f"snapshot_{self.num_timesteps}"
            path = os.path.join(self.model_dir, "a2_checkpoints", f"{snapshot_name}.zip")
            self.model.save(path)
            
            if self.verbose:
                log.info(f"[SelfPlay A2] Saved snapshot to {path}")
            
            # Thêm model mới vào pool phối hợp
            self.pool_manager.pool_phoi_hop.append(path)
            
            ckpt_dir = os.path.join(self.model_dir, "a2_checkpoints")
            existing = [f for f in os.listdir(ckpt_dir) if f.startswith("snapshot_") and f.endswith(".zip")]
            N = len(existing)
            interval = (N + 1) * 500_000
            self.next_snapshot_at += interval

        rew = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
        if rew is not None:
            if rew > self.best_reward:
                self.best_reward = rew
                self.model.save(self.best_path)
                if self.verbose:
                    log.info(f"[SelfPlay A2] New best: {rew:.3f} -> {self.best_path}.zip")

            if rew >= self.target_reward:
                log.info(f"[SelfPlay A2] Target reached! ep_rew_mean={rew:.3f}")
                return False

        return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--initial-model", default="models/a1_checkpoints/snapshot_15000000.zip", help="Path to initial model")
    p.add_argument("--steps",         default=100_000_000, type=int)
    p.add_argument("--resume",        default=None,        help="Resume from a previous A2 checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs("models/a2_checkpoints", exist_ok=True)

    pool_manager = PoolManager()

    # Load all A1 models for 1v1 pool
    a1_full = sorted(glob.glob("models/a1_checkpoints/snapshot_*.zip"))
    pool_manager.a1_full_pool = a1_full
    log.info(f"[SelfPlay A2] Found {len(a1_full)} A1 models for 1v1 pool.")

    # Initialize pool_ca_nhan_teammate and pool_ca_nhan_opponent
    pool_manager.pool_ca_nhan_teammate = []
    pool_manager.pool_ca_nhan_opponent = []
    
    import re
    for path in a1_full:
        basename = os.path.basename(path)
        m = re.search(r"snapshot_(\d+)", basename)
        if m:
            step = int(m.group(1))
            if step <= 15_000_000:
                pool_manager.pool_ca_nhan_teammate.append(path)
            else:
                pool_manager.pool_ca_nhan_opponent.append(path)
                
    if os.path.exists("models/a1_final.zip"):
        pool_manager.pool_ca_nhan_opponent.append("models/a1_final.zip")
        
    log.info(f"[SelfPlay A2] Initialized pool_ca_nhan_teammate with {len(pool_manager.pool_ca_nhan_teammate)} models.")
    log.info(f"[SelfPlay A2] Initialized pool_ca_nhan_opponent with {len(pool_manager.pool_ca_nhan_opponent)} models.")

    # Load existing A2 models into pool_phoi_hop
    a2_models = sorted(glob.glob("models/a2_checkpoints/snapshot_*.zip"))
    pool_manager.pool_phoi_hop = a2_models
    log.info(f"[SelfPlay A2] Initialized pool_phoi_hop with {len(a2_models)} models.")

    def make_env():
        return SelfPlayEnvA2(pool_manager=pool_manager, phase='A2', n_agents=3, p_1v1=0.2)

    vec_env = DummyVecEnv([make_env for _ in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    if args.resume:
        log.info(f"[SelfPlay A2] Resuming from {args.resume}")
        custom_objects = PPO_PARAMS.copy()
        if "policy_kwargs" in custom_objects:
            del custom_objects["policy_kwargs"]
        model = PPO.load(args.resume, env=vec_env, custom_objects=custom_objects)
        model.set_env(vec_env)
    else:
        log.info(f"[SelfPlay A2] Loading pretrained weights from {args.initial_model}")
        custom_objects = PPO_PARAMS.copy()
        if "policy_kwargs" in custom_objects:
            del custom_objects["policy_kwargs"]
        model = PPO.load(args.initial_model, env=vec_env, custom_objects=custom_objects)
        model.set_env(vec_env)
        
    is_resume = bool(args.resume)

    callback = SelfPlayCallback(
        pool_manager=pool_manager,
        target_reward=TARGET_REWARD,
        model_dir="models/",
        envs=vec_env,
        verbose=1
    )

    if is_resume:
        remaining_steps = max(args.steps - model.num_timesteps, 0)
        log.info(f"[SelfPlay A2] Resuming from step {model.num_timesteps:,} — {remaining_steps:,} steps remaining")
    else:
        remaining_steps = args.steps

    log.info(f"[SelfPlay A2] Starting — {N_ENVS} envs, {remaining_steps:,} steps to go")

    model.learn(
        total_timesteps      = remaining_steps,
        callback             = callback,
        reset_num_timesteps  = not is_resume,
        log_interval         = 250_000 // (PPO_PARAMS["n_steps"] * N_ENVS),
        progress_bar         = True,
    )

    model.save("models/a2_pfsp_final")
    log.info("[SelfPlay A2] Done -> models/a2_pfsp_final.zip")
