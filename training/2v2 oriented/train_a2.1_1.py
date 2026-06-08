"""
train_a3.1.py — Finetune trên Map 3v3 To (từ a1.1_final).

Chiến lược:
  • Finetune từ a1.1_final (1v1 → 3v3 to).
  • 250k step đầu: chỉ chơi với bot (chưa có self-play).
  • Sau 250k: self-play dynamic interval = 250k * (N_pool + 1).
  • Snapshot cuối tại 6,000,000 steps.
  • Snapshot → models/2v2 oriented/a3_1_checkpoints/
  • Final   → models/2v2 oriented/a3_1_final.zip

Usage:
    python "training/2v2 oriented/train_a3.1.py"
    python "training/2v2 oriented/train_a3.1.py" --steps 6000000
    python "training/2v2 oriented/train_a3.1.py" --resume "models/2v2 oriented/a3_1_checkpoints/snapshot_1000000.zip"
    python "training/2v2 oriented/train_a3.1.py" --a11-model "models/1v1 oriented/a1.1_final.zip"
"""

import argparse
import gc
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from training.env import HaxballCurriculumEnv
from utils.model_logger import get_model_logger

log = get_model_logger("a2.1_1")

# ── Config ─────────────────────────────────────────────────────────────────────
N_ENVS        = 8
TARGET_REWARD = 15.0
BOT_TYPES     = ('Wanderer', 'Pazzo', 'Random', 'Static')

TOTAL_STEPS   = 6_000_000
BOT_ONLY_STEPS = 250_000          # 250k step đầu chỉ đánh bot
SNAP_INTERVAL_BASE = 250_000      # snapshot mỗi 250k * (N_pool + 1)
FINAL_SNAP_STEP   = 6_000_000    # snapshot bắt buộc tại step cuối

PPO_PARAMS = dict(
    n_steps      = 2048,
    batch_size   = 512,
    n_epochs     = 10,
    gamma        = 0.99,
    gae_lambda   = 0.95,
    ent_coef     = 0.01,
    learning_rate= 5e-5,          # LR nhỏ cho finetune
    clip_range   = 0.1,           # clip nhỏ cho finetune
    vf_coef      = 0.5,
    max_grad_norm= 0.5,
    policy_kwargs= dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
)

CKPT_DIR   = "models/2v2 oriented/a3_1_checkpoints"
FINAL_PATH = "models/2v2 oriented/a3_1_final"

A11_MODEL_DEFAULT = "models/1v1 oriented/a1.1_final.zip"


# ── Self-Play Opponent Manager ─────────────────────────────────────────────────
class OpponentManager:
    def __init__(self):
        self.opponents = {}        # name → policy
        self.agent_pool = []
        self.bot_pool = list(BOT_TYPES)
        self.terminated_pool = []

        self.agent_idx = 0
        self.terminated_idx = 0

        # Stats per opponent: wins = số lần snapshot thắng (agent thua)
        self.stats = {}  # name → {'wins': 0, 'total': 0}

    def add_opponent(self, name, model_path):
        log.info(f"[SelfPlay] Loading opponent '{name}' from {model_path}...")
        policy = PPO.load(model_path, device='cpu')
        self.opponents[name] = policy
        self.stats[name] = {'wins': 0, 'total': 0}
        self.agent_pool.append(name)

    def update_result(self, name, agent_won):
        if name in self.stats:
            self.stats[name]['total'] += 1
            if not agent_won:
                self.stats[name]['wins'] += 1

            h = self.stats[name]
            if name in self.agent_pool and h['total'] >= 20:
                snapshot_winrate = h['wins'] / h['total']
                if snapshot_winrate < 0.05:
                    log.info(
                        f"[SelfPlay] Opponent '{name}' winrate {snapshot_winrate:.3f} < 0.05 "
                        f"after {h['total']} matches → moving to terminated pool."
                    )
                    if name in self.agent_pool:
                        self.agent_pool.remove(name)
                    self.terminated_pool.append(name)
                    self.agent_idx = 0
                    self.terminated_idx = 0

    def sample_opponent(self, current_step=0):
        """
        - current_step < BOT_ONLY_STEPS → luôn trả về bot (chưa self-play).
        - Sau đó: vòng tròn giữa agent_pool và 1 bot slot.
        """
        if current_step < BOT_ONLY_STEPS:
            return self._sample_bot(force_any=True)

        num_agents = len(self.agent_pool)
        total_entities = num_agents + 1  # +1 cho bot slot

        idx = self.agent_idx % total_entities
        self.agent_idx = (self.agent_idx + 1) % total_entities

        if idx == num_agents:
            return self._sample_bot(force_any=False)
        else:
            chosen_name = self.agent_pool[idx]
            return self.opponents[chosen_name], chosen_name

    def _sample_bot(self, force_any=False):
        """Chọn bot ngẫu nhiên. force_any=True: bỏ Wanderer nếu chưa có trong pool."""
        r = np.random.rand()
        if force_any:
            # Không dùng Wanderer ở giai đoạn bot-only ban đầu
            if r < 0.33:
                bot_name = 'Pazzo'
            elif r < 0.66:
                bot_name = 'Random'
            else:
                bot_name = 'Static'
        else:
            if r < 0.82:
                bot_name = 'Wanderer'
            elif r < 0.88:
                bot_name = 'Pazzo'
            elif r < 0.94:
                bot_name = 'Random'
            else:
                bot_name = 'Static'
        return None, bot_name


# ── Self-Play Environment Wrapper (map 3v3 to) ────────────────────────────────
class SelfPlayEnv(HaxballCurriculumEnv):
    def __init__(self, opponent_manager, **kwargs):
        super().__init__(phase='A2.1_1', **kwargs)
        self.opponent_manager = opponent_manager
        self.current_opponent_name = None

    def _reset_positions(self):
        super()._reset_positions()

        if self.episode_type == 'opponent':
            current_step = getattr(self, 'total_timesteps_elapsed', 0)
            policy, name = self.opponent_manager.sample_opponent(current_step)
            self.current_opponent_name = name
            if name in BOT_TYPES:
                self.opponent_type = name
                self.opponent_policy = None
            else:
                self.opponent_type = 'Trained'
                self.opponent_policy = policy

    def step(self, action):
        opponent_name = self.current_opponent_name
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated or truncated:
            agent_score = self.scores[self.team_id - 1]
            opp_score   = self.scores[2 - self.team_id]

            if self.episode_type == 'opponent' and opponent_name is not None:
                if agent_score != opp_score:
                    agent_won = (agent_score > opp_score)
                    self.opponent_manager.update_result(opponent_name, agent_won)

        return obs, reward, terminated, truncated, info


# ── Callback ──────────────────────────────────────────────────────────────────
class A31Callback(BaseCallback):
    def __init__(self, opponent_manager, target_reward, model_dir, envs, verbose=1):
        super().__init__(verbose)
        self.opponent_manager = opponent_manager
        self.target_reward    = target_reward
        self.model_dir        = model_dir
        self.envs             = envs
        self.best_reward      = -np.inf
        self.best_path        = os.path.join(model_dir, "a3_1_best")
        self._compute_next_snapshot(from_step=0)
        self._final_snap_saved = False

    def _compute_next_snapshot(self, from_step):
        """next_snapshot_at = from_step + 250k * (N_pool + 1), không vượt FINAL_SNAP_STEP."""
        N = len(self.opponent_manager.agent_pool)
        interval = SNAP_INTERVAL_BASE * (N + 1)
        candidate = from_step + interval
        self.next_snapshot_at = min(candidate, FINAL_SNAP_STEP)
        log.info(
            f"[A2.1_1] Next snapshot tại step {self.next_snapshot_at:,} "
            f"(interval={interval:,}, pool_size={N})"
        )

    def _on_training_start(self) -> None:
        N = len(self.opponent_manager.agent_pool)
        if N > 0:
            # Resume: tính lại từ current step
            self._compute_next_snapshot(from_step=self.num_timesteps)
            log.info(
                f"[A2.1_1] Resume: {N} existing snapshots → "
                f"next snapshot tại step {self.next_snapshot_at:,}"
            )
        else:
            # Fresh start: snapshot đầu tiên sau 250k (bot-only phase)
            self.next_snapshot_at = BOT_ONLY_STEPS + SNAP_INTERVAL_BASE  # = 500k
            log.info(
                f"[A2.1_1] Fresh finetune từ a1.1_final. "
                f"Bot-only phase: {BOT_ONLY_STEPS:,} steps. "
                f"Snapshot đầu tiên tại step {self.next_snapshot_at:,}."
            )

    def _save_snapshot(self):
        """Lưu snapshot tại current step và đăng ký vào opponent pool."""
        step = self.num_timesteps
        snapshot_name = f"snapshot_{step}"
        path = os.path.join(self.model_dir, "a3_1_checkpoints", f"{snapshot_name}.zip")
        self.model.save(path)
        log.info(f"[A2.1_1] Saved snapshot → {path}")

        self.opponent_manager.add_opponent(snapshot_name, path)

        # Log stats
        log.info("\n--- SelfPlay Opponent Winrates (A2.1_1) ---")
        log.info("Agent Pool: " + ", ".join(self.opponent_manager.agent_pool))
        log.info("Bot Pool: "   + ", ".join(self.opponent_manager.bot_pool))
        log.info("Terminated: " + ", ".join(self.opponent_manager.terminated_pool))
        log.info("Stats:")
        for name, h in self.opponent_manager.stats.items():
            if h['total'] > 0:
                wr = h['wins'] / h['total']
                log.info(f"  {name}: {wr*100:.1f}% ({h['wins']}/{h['total']})")
            else:
                log.info(f"  {name}: No matches yet")
        log.info("------------------------------------------")

        return step

    def _on_step(self) -> bool:
        step = self.num_timesteps

        # 1. Sync timesteps vào env để curriculum đọc đúng
        for env in self.envs.envs:
            env.total_timesteps_elapsed = step

        # 2. Snapshot theo lịch
        while self.next_snapshot_at <= step:
            self._save_snapshot()
            if step >= FINAL_SNAP_STEP:
                self._final_snap_saved = True
                break
            self._compute_next_snapshot(from_step=step)

        # 3. Best reward tracking
        rew = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
        if rew is not None:
            if rew > self.best_reward:
                self.best_reward = rew
                self.model.save(self.best_path)
                if self.verbose:
                    log.info(f"[A2.1_1] New best: {rew:.3f} → {self.best_path}.zip")

            if rew >= self.target_reward:
                log.info(f"[A2.1_1] Target reached! ep_rew_mean={rew:.3f}")
                return False

        return True


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="train_a3.1 — finetune từ a1.1_final trên map 3v3 to"
    )
    p.add_argument(
        "--a11-model",
        default=A11_MODEL_DEFAULT,
        help="Path to a1.1_final model (weights khởi đầu finetune)"
    )
    p.add_argument("--steps",  default=TOTAL_STEPS, type=int, help="Tổng steps (default: 6,000,000)")
    p.add_argument("--resume", default=None, help="Resume từ checkpoint a2.1_1 (.zip)")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)

    # 1. Opponent Manager
    opponent_manager = OpponentManager()

    def make_env():
        return SelfPlayEnv(opponent_manager=opponent_manager)

    vec_env = DummyVecEnv([make_env for _ in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    custom_objects = PPO_PARAMS.copy()
    if "policy_kwargs" in custom_objects:
        del custom_objects["policy_kwargs"]

    # 2. Load model
    if args.resume:
        log.info(f"[A2.1_1] Resuming từ {args.resume}")
        model = PPO.load(args.resume, env=vec_env, custom_objects=custom_objects)
        model.set_env(vec_env)
        is_resume = True
    else:
        if not os.path.exists(args.a11_model):
            log.error(f"[A2.1_1] Không tìm thấy model: {args.a11_model}")
            sys.exit(1)
        log.info(f"[A2.1_1] Finetune từ a1.1_final: {args.a11_model}")
        model = PPO.load(args.a11_model, env=vec_env, custom_objects=custom_objects)
        model.set_env(vec_env)
        is_resume = False

    # 3. On resume: reload existing snapshots
    if is_resume:
        existing = sorted(
            [f for f in os.listdir(CKPT_DIR) if f.startswith("snapshot_") and f.endswith(".zip")],
            key=lambda f: int(f.replace("snapshot_", "").replace(".zip", ""))
        )
        log.info(f"[A2.1_1] Found {len(existing)} snapshot(s) → reloading into opponent pool...")
        for fname in existing:
            snap_name = fname.replace(".zip", "")
            snap_path = os.path.join(CKPT_DIR, fname)
            opponent_manager.add_opponent(snap_name, snap_path)
            log.info(f"[A2.1_1] Reloaded opponent '{snap_name}'")

    # 4. Callback
    callback = A31Callback(
        opponent_manager=opponent_manager,
        target_reward=TARGET_REWARD,
        model_dir="models/2v2 oriented/",
        envs=vec_env,
        verbose=1,
    )

    # 5. Remaining steps
    if is_resume:
        remaining_steps = max(args.steps - model.num_timesteps, 0)
        log.info(
            f"[A2.1_1] Resuming từ step {model.num_timesteps:,} — "
            f"{remaining_steps:,} steps còn lại"
        )
    else:
        remaining_steps = args.steps

    log.info(
        f"[A2.1_1] Bắt đầu finetune — {N_ENVS} envs | {remaining_steps:,} steps | "
        f"Map 3v3 to | Bot-only {BOT_ONLY_STEPS:,} steps → Self-play dynamic 250k*(N+1)"
    )

    # 6. Train
    model.learn(
        total_timesteps     = remaining_steps,
        callback            = callback,
        reset_num_timesteps = not is_resume,
        log_interval        = SNAP_INTERVAL_BASE // (PPO_PARAMS["n_steps"] * N_ENVS),
        progress_bar        = True,
    )

    # 7. Save final
    model.save(FINAL_PATH)
    log.info(f"[A2.1_1] Done → {FINAL_PATH}.zip")

    gc.collect()
