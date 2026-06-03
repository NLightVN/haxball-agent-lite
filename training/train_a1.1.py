"""
train_a1.1.py — Solo (1v1) Self-Play on the 2v2 Map.

Sử dụng map 2v2, agent a1_final solo 1v1 với toàn bộ pool a1.
  • 1.000.000 step đầu: lưu snapshot a1.1_snapshot_1000000, thêm vào pool.
  • 1.000.000 step tiếp theo: lưu a1.1_snapshot_2000000, dừng.

Usage:
    python training/train_a1.1.py
    python training/train_a1.1.py --resume models/a1.1_checkpoints/a1.1_snapshot_1000000.zip
"""

import argparse
import glob
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from training.env import HaxballCurriculumEnv
from utils.model_logger import get_model_logger

log = get_model_logger("a1.1")

# ── Config ─────────────────────────────────────────────────────────────────────
N_ENVS        = 8
TOTAL_STEPS   = 2_000_000   # 1M + 1M
SNAPSHOT_INTERVAL = 1_000_000

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

# ── Opponent Pool Manager ─────────────────────────────────────────────────────
class PoolManager:
    """
    Quản lý pool đối thủ: tất cả snapshots của a1 và a1_final,
    sau đó cộng thêm các snapshot a1.1 khi được lưu.
    """
    def __init__(self):
        self.pool = []          # list of paths (strings)
        self._loaded = {}       # path -> PPO policy (lazy load)
        self._rr_idx = 0        # round-robin index

    def add(self, path: str):
        if path not in self.pool:
            self.pool.append(path)
            log.info(f"[Pool] Added opponent: {path} (pool size={len(self.pool)})")

    def get_policy(self, path: str):
        if path not in self._loaded:
            log.info(f"[Pool] Loading policy: {path}")
            self._loaded[path] = PPO.load(path, device='cpu')
        return self._loaded[path]

    def sample_opponent(self):
        """Round-robin sampling qua toàn bộ pool."""
        if not self.pool:
            return None
        path = self.pool[self._rr_idx % len(self.pool)]
        self._rr_idx = (self._rr_idx + 1) % len(self.pool)
        return self.get_policy(path)


# ── Self-Play Environment (2v2 map, 1v1 match) ───────────────────────────────
class SelfPlayEnvA1_1(HaxballCurriculumEnv):
    """
    Dùng map 2v2 (phase='A2') nhưng luôn thi đấu 1v1 (p_1v1=1.0).
    Đối thủ được lấy round-robin từ pool.
    """
    def __init__(self, pool_manager: PoolManager, **kwargs):
        # phase='A2'  → tải map_2v2.json kích thước sân/gôn
        # p_1v1=1.0   → luôn spawn 1 agent + 1 opponent (không có đồng đội)
        super().__init__(phase='A2', p_1v1=1.0, **kwargs)
        self.pool_manager = pool_manager
        self._current_opp_policy = None

    def _reset_positions(self):
        # Gọi base để setup map kích thước, spawn agent + opp (is_1v1=True branch)
        super()._reset_positions()

        # Override opponent policy từ pool
        if self.episode_type == 'opponent':
            policy = self.pool_manager.sample_opponent()
            self.opponent_type = 'Trained'
            # Với is_1v1 branch, base class dùng self.opponent_policies[0]
            if policy is not None:
                self.opponent_policies = [policy]
            else:
                # Pool rỗng: để opponent đứng im
                self.opponent_policies = []
            self._current_opp_policy = policy


# ── Callback: snapshot sau mỗi SNAPSHOT_INTERVAL bước, dừng sau 2 snapshots ──
class A1_1Callback(BaseCallback):
    def __init__(self, pool_manager: PoolManager, model_dir: str, envs, verbose=1):
        super().__init__(verbose)
        self.pool_manager = pool_manager
        self.model_dir = model_dir
        self.envs = envs
        self.ckpt_dir = os.path.join(model_dir, "a1.1_checkpoints")
        self.snapshots_saved = 0
        self.next_snapshot_at = SNAPSHOT_INTERVAL
        self.best_reward = -np.inf
        self.best_path = os.path.join(model_dir, "a1.1_best")

    def _on_training_start(self) -> None:
        # Nếu resume: đếm số snapshot đã lưu để đặt lại next_snapshot_at
        existing = sorted(glob.glob(os.path.join(self.ckpt_dir, "a1.1_snapshot_*.zip")))
        self.snapshots_saved = len(existing)
        self.next_snapshot_at = (self.snapshots_saved + 1) * SNAPSHOT_INTERVAL
        log.info(
            f"[A1.1] Training start — "
            f"{self.snapshots_saved} snapshot(s) đã có, "
            f"next snapshot tại step {self.next_snapshot_at:,}"
        )

    def _on_step(self) -> bool:
        # Đồng bộ timestep cho curriculum env (dù A1.1 không dùng curriculum riêng)
        for env in self.envs.envs:
            env.total_timesteps_elapsed = self.num_timesteps

        # Kiểm tra có đến mốc snapshot không
        while self.next_snapshot_at <= self.num_timesteps:
            self.snapshots_saved += 1
            snapshot_name = f"a1.1_snapshot_{self.num_timesteps}"
            path = os.path.join(self.ckpt_dir, f"{snapshot_name}.zip")
            self.model.save(path)
            log.info(f"[A1.1] Snapshot #{self.snapshots_saved} saved → {path}")

            # Thêm snapshot mới vào pool → agent sẽ solo với chính snapshot của mình
            self.pool_manager.add(path)

            # Log pool hiện tại
            log.info(f"[A1.1] Pool hiện tại ({len(self.pool_manager.pool)} đối thủ):")
            for p in self.pool_manager.pool:
                log.info(f"  {os.path.basename(p)}")

            # Dừng sau khi đã lưu 2 snapshots (tổng 2M steps)
            if self.snapshots_saved >= 2:
                log.info("[A1.1] Đã đủ 2 snapshots (2M steps). Dừng training.")
                return False

            self.next_snapshot_at += SNAPSHOT_INTERVAL

        # Theo dõi best reward
        rew = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
        if rew is not None and rew > self.best_reward:
            self.best_reward = rew
            self.model.save(self.best_path)
            if self.verbose:
                log.info(f"[A1.1] New best reward: {rew:.3f} → {self.best_path}.zip")

        return True


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="train_a1.1 — 1v1 self-play trên map 2v2")
    p.add_argument(
        "--initial-model",
        default="models/a1_final.zip",
        help="Model khởi đầu (mặc định: models/a1_final.zip)"
    )
    p.add_argument(
        "--resume",
        default=None,
        help="Tiếp tục từ checkpoint a1.1 (vd: models/a1.1_checkpoints/a1.1_snapshot_1000000.zip)"
    )
    p.add_argument("--steps", default=TOTAL_STEPS, type=int, help="Tổng số steps")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    os.makedirs("models/a1.1_checkpoints", exist_ok=True)

    # 1. Khởi tạo pool với toàn bộ snapshots a1 + a1_final
    pool = PoolManager()

    a1_snapshots = sorted(
        glob.glob("models/a1_checkpoints/snapshot_*.zip"),
        key=lambda f: int(
            os.path.basename(f).replace("snapshot_", "").replace(".zip", "")
        )
    )
    log.info(f"[A1.1] Tìm thấy {len(a1_snapshots)} snapshot(s) a1.")
    for path in a1_snapshots:
        pool.add(path)

    if os.path.exists("models/a1_final.zip"):
        pool.add("models/a1_final.zip")
    else:
        log.warning("[A1.1] Không tìm thấy models/a1_final.zip!")

    # Nếu resume: nạp lại các snapshot a1.1 đã lưu vào pool
    a1_1_snapshots = sorted(
        glob.glob("models/a1.1_checkpoints/a1.1_snapshot_*.zip"),
        key=lambda f: int(
            os.path.basename(f).replace("a1.1_snapshot_", "").replace(".zip", "")
        )
    )
    if a1_1_snapshots:
        log.info(f"[A1.1] Tìm thấy {len(a1_1_snapshots)} snapshot(s) a1.1 đã lưu — nạp vào pool.")
        for path in a1_1_snapshots:
            pool.add(path)

    log.info(f"[A1.1] Pool khởi tạo với {len(pool.pool)} đối thủ.")

    # 2. Tạo môi trường
    def make_env():
        return SelfPlayEnvA1_1(pool_manager=pool)

    vec_env = DummyVecEnv([make_env for _ in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    # 3. Tải model
    custom_objects = {k: v for k, v in PPO_PARAMS.items() if k != "policy_kwargs"}

    if args.resume:
        log.info(f"[A1.1] Resuming từ {args.resume}")
        model = PPO.load(args.resume, env=vec_env, custom_objects=custom_objects)
        model.set_env(vec_env)
        is_resume = True
    else:
        log.info(f"[A1.1] Tải model khởi đầu từ {args.initial_model}")
        model = PPO.load(args.initial_model, env=vec_env, custom_objects=custom_objects)
        model.set_env(vec_env)
        is_resume = False

    # 4. Callback
    callback = A1_1Callback(
        pool_manager=pool,
        model_dir="models/",
        envs=vec_env,
        verbose=1
    )

    # 5. Tính số bước còn lại
    if is_resume:
        remaining_steps = max(args.steps - model.num_timesteps, 0)
        log.info(
            f"[A1.1] Resume từ step {model.num_timesteps:,} — "
            f"còn {remaining_steps:,} steps"
        )
    else:
        remaining_steps = args.steps

    log.info(
        f"[A1.1] Bắt đầu training — {N_ENVS} envs, {remaining_steps:,} steps, "
        f"map 2v2, {len(pool.pool)} đối thủ trong pool"
    )

    # 6. Train
    model.learn(
        total_timesteps     = remaining_steps,
        callback            = callback,
        reset_num_timesteps = not is_resume,
        log_interval        = 250_000 // (PPO_PARAMS["n_steps"] * N_ENVS),
        progress_bar        = True,
    )

    # 7. Lưu final (nếu training kết thúc tự nhiên, không bị dừng bởi callback)
    model.save("models/a1.1_final")
    log.info("[A1.1] Done → models/a1.1_final.zip")
