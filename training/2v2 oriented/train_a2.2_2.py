"""
train_a3.2.py — 2v2 Self-Play Training with PFSP and asymmetric 2v1 matches.

Curriculum:
  - 10% matches (1 env): 1v1 on 3v3 map with a2.1_1 snapshots.
  - Dynamic matches (7 envs):
      - 2v1 probability: max(90% / (N_a3_2 + 1), 15%)
      - 2v2 probability: the rest
      - 5% chaos mode: 2 random a2.1_1 opponents (starts after 5M steps)
      - 2v2 self play: 70% PFSP, 30% uniform.

Opponent Pool: a2.1_1 snapshots.
Teammate: current_model (agent being trained).
"""

import argparse
import glob
import os
import sys
import msvcrt
from collections import deque

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from training.env import HaxballCurriculumEnv
from utils.model_logger import get_model_logger

log = get_model_logger("a2.2_2")

# ── Config ─────────────────────────────────────────────────────────────────────
N_ENVS            = 8
TARGET_REWARD     = 15.0

# PFSP params
PFSP_MIN_GAMES   = 20
PFSP_FLOOR       = 0.05

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

# ── Opponent Pool with PFSP ───────────────────────────────────────────────────
class OpponentPoolA3_2:
    def __init__(self, min_games: int = PFSP_MIN_GAMES, floor: float = PFSP_FLOOR):
        self.pool_a3_1: list[str] = []
        self.pool_a3_2: list[str] = []
        self._loaded: dict[str, PPO] = {}
        self.stats: dict[str, deque] = {} 
        self.min_games = min_games
        self.floor = floor

    @property
    def num_a3_2(self):
        return len(self.pool_a3_2)

    def add(self, path: str):
        if 'a3_1_checkpoints' in path:
            if path not in self.pool_a3_1:
                self.pool_a3_1.append(path)
                log.info(f"[Pool A2.1_1] + {os.path.basename(path)}  (total={len(self.pool_a3_1)})")
        else:
            if path not in self.pool_a3_2:
                self.pool_a3_2.append(path)
                self.stats[path] = deque(maxlen=500) 
                log.info(f"[Pool A2.2_2] + {os.path.basename(path)}  (total={len(self.pool_a3_2)})")

    def get_policy(self, path: str) -> PPO:
        if path not in self._loaded:
            log.info(f"[Pool] Loading {path}...")
            self._loaded[path] = PPO.load(path, device='cpu')
        return self._loaded[path]

    def update_result(self, path: str, agent_won: bool):
        if path not in self.stats:
            return
        self.stats[path].append(agent_won)

    def _weight(self, path: str) -> float:
        history = self.stats.get(path, [])
        total = len(history)
        if total < self.min_games:
            return 1.0  
        wins_for_snapshot = sum(1 for agent_won in history if not agent_won)
        snapshot_wr = wins_for_snapshot / total  
        agent_wr    = 1.0 - snapshot_wr        
        return max(4.0 * agent_wr * (1.0 - agent_wr), self.floor)

    def sample(self) -> tuple[PPO | None, str | None]:
        if not self.pool_a3_2: return None, None
        weights = np.array([self._weight(p) for p in self.pool_a3_2], dtype=np.float64)
        weights /= weights.sum()
        path = np.random.choice(self.pool_a3_2, p=weights)
        return self.get_policy(path), path

    def sample_uniform(self) -> tuple[PPO | None, str | None]:
        if not self.pool_a3_2: return None, None
        path = np.random.choice(self.pool_a3_2)
        return self.get_policy(path), path

    def sample_uniform_a3_1(self) -> tuple[PPO | None, str | None]:
        if not self.pool_a3_1: return None, None
        path = np.random.choice(self.pool_a3_1)
        return self.get_policy(path), path

    def log_stats(self):
        log.info(f"\n{'-'*60}")
        log.info(f"  PFSP Opponent Pool A2.2_2 — {len(self.pool_a3_2)} agents")
        log.info(f"{'-'*60}")
        if not self.pool_a3_2:
            log.info("  (Empty pool)")
            log.info(f"{'-'*60}")
            return
            
        weights = {p: self._weight(p) for p in self.pool_a3_2}
        total_w = sum(weights.values()) or 1.0
        for path in self.pool_a3_2:
            history = self.stats[path]
            total = len(history)
            wins_for_snapshot = sum(1 for agent_won in history if not agent_won)
            name = os.path.basename(path).replace('.zip', '')
            w = weights[path] / total_w * 100
            if total >= self.min_games:
                snap_wr  = wins_for_snapshot / total    
                agent_wr = 1.0 - snap_wr
                log.info(
                    f"  {name:<40} "
                    f"opp_wr={snap_wr*100:5.1f}%  agent_wr={agent_wr*100:5.1f}%  "
                    f"({wins_for_snapshot:>4}/{total:<4})  sel={w:.1f}%"
                )
            else:
                log.info(f"  {name:<40} opp_wr=  ?      agent_wr=  ?     ({wins_for_snapshot:>4}/{total:<4}  <{self.min_games})  sel={w:.1f}%")
        log.info(f"{'-'*60}")


# ── Self-Play Environment ─────────────────────────────────────────────────────
class SelfPlayEnvA3_2(HaxballCurriculumEnv):
    def __init__(self, opponent_pool: OpponentPoolA3_2, fixed_mode: str = None, **kwargs):
        super().__init__(phase='A2.2_2', fixed_mode=fixed_mode, **kwargs)
        self.opponent_pool = opponent_pool
        self._current_opp_paths: list[str] = []   

    def _reset_positions(self):
        super()._reset_positions()

        if self.episode_type == 'opponent':
            self.opponent_type = 'Trained'
            
            # Teammate is controlled by a snapshot of itself, not shared policy
            if self.opponent_pool.num_a3_2 > 0:
                tm_pol, _ = self.opponent_pool.sample_uniform()
            else:
                tm_pol, _ = self.opponent_pool.sample_uniform_a3_1()
            self.teammate_policy = tm_pol
            
            
            if getattr(self, 'is_1v1', False):
                # 1v1 on 3v3 map with a2.1_1 snapshot
                pol, path = self.opponent_pool.sample_uniform_a3_1()
                self._current_opp_paths = [] 
                self.opponent_policies = [pol] if pol else []
            elif getattr(self, 'is_2v1', False):
                # 2v1 on 3v3 map with a2.1_1 snapshot
                pol, path = self.opponent_pool.sample_uniform_a3_1()
                self._current_opp_paths = [] 
                self.opponent_policies = [pol] if pol else []
            elif getattr(self, 'is_2v2', False):
                if getattr(self, 'is_chaos', False):
                    pol1, path1 = self.opponent_pool.sample_uniform_a3_1()
                    pol2, path2 = self.opponent_pool.sample_uniform_a3_1()
                    self._current_opp_paths = [path1, path2] if (path1 and path2) else []
                    self.opponent_policies = [pol1, pol2] if (pol1 and pol2) else []
                else:
                    # 2v2 self play: 70% PFSP, 30% uniform
                    if np.random.rand() < 0.7:
                        pol, path = self.opponent_pool.sample()
                    else:
                        pol, path = self.opponent_pool.sample_uniform()
                    
                    # Assign to both opponents
                    self._current_opp_paths = [path] if path else []
                    self.opponent_policies = [pol, pol] if pol else []

    def step(self, action):
        opp_paths = self._current_opp_paths
        obs, reward, terminated, truncated, info = super().step(action)

        if (terminated or truncated) and opp_paths:
            agent_score = self.scores[self.team_id - 1]
            opp_score   = self.scores[2 - self.team_id]
            if agent_score != opp_score: 
                agent_won = agent_score > opp_score
                for p in opp_paths:
                    self.opponent_pool.update_result(p, agent_won)

        return obs, reward, terminated, truncated, info


# ── Callback ───────────────────────────────────────────────────────────────────
class SelfPlayCallbackA3_2(BaseCallback):
    def __init__(self, opponent_pool: OpponentPoolA3_2, model_dir: str, envs, verbose=1):
        super().__init__(verbose)
        self.opponent_pool = opponent_pool
        self.model_dir = model_dir
        self.envs = envs
        self.ckpt_dir = os.path.join(model_dir, "a3_2_checkpoints")
        self.best_reward = -np.inf
        self.best_path = os.path.join(model_dir, "a3_2_best")
        self.next_snapshot_at = 3_000_000
        self.last_log_step = 0

    def _get_next_snapshot_step(self, N_in_pool: int) -> int:
        accumulated = sum((k + 1) * 500_000 for k in range(N_in_pool))
        return int(accumulated + (N_in_pool + 1) * 500_000)

    def _get_current_pool_size(self) -> int:
        return self.opponent_pool.num_a3_2

    def _on_training_start(self) -> None:
        N = self._get_current_pool_size()
        self.next_snapshot_at = self._get_next_snapshot_step(N)
        if N > 0:
            log.info(f"[A2.2_2] Resume: {N} snapshot(s) trong pool, next snapshot tai step {self.next_snapshot_at:,}")
        else:
            log.info(f"[A2.2_2] Next snapshot tai step {self.next_snapshot_at:,}")

    def _on_step(self) -> bool:
        for env in self.envs.envs:
            env.total_timesteps_elapsed = self.num_timesteps
            env.current_model = self.model

        if msvcrt.kbhit():
            try:
                key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                if key == 's':
                    snap_name = f"emergency_nopool_{self.num_timesteps}"
                    path = os.path.join(self.ckpt_dir, f"{snap_name}.zip")
                    self.model.save(path)
                    log.info(f"[A2.2_2] KHAN CAP (Phím S): Đã lưu snapshot KHÔNG vào pool -> {path}")
                elif key == 'p':
                    snap_name = f"snapshot_{self.num_timesteps}_emergency_pool"
                    path = os.path.join(self.ckpt_dir, f"{snap_name}.zip")
                    self.model.save(path)
                    self.opponent_pool.add(path)
                    
                    N = self._get_current_pool_size()
                    self.next_snapshot_at = self._get_next_snapshot_step(N)
                    log.info(f"[A2.2_2] KHAN CAP (Phím P): Đã lưu snapshot VÀO POOL -> {path}")
            except Exception:
                pass

        if self.num_timesteps - self.last_log_step >= 250_000:
            self.opponent_pool.log_stats()
            self.last_log_step = self.num_timesteps

        while self.next_snapshot_at <= self.num_timesteps:
            snap_name = f"snapshot_{self.num_timesteps}"
            path = os.path.join(self.ckpt_dir, f"{snap_name}.zip")
            self.model.save(path)
            log.info(f"[A2.2_2] Saved snapshot -> {path}")

            # Do not add to a2.1_1 pool (since pool is strict a2.1_1 snapshots), 
            # unless we want to self-play against new a2.2_2 snapshots. 
            # The prompt said "doi thu la cac snapshot trong pool 3.1".
            # We will add it to the pool anyway for true self-play progression.
            self.opponent_pool.add(path)

            N = self._get_current_pool_size()
            self.next_snapshot_at = self._get_next_snapshot_step(N)

        rew = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
        if rew is not None:
            if rew > self.best_reward:
                self.best_reward = rew
                self.model.save(self.best_path)
                if self.verbose:
                    log.info(f"[A2.2_2] New best: {rew:.3f} -> {self.best_path}.zip")
            if rew >= TARGET_REWARD:
                log.info(f"[A2.2_2] Target reached! ep_rew_mean={rew:.3f}")
                return False

        return True


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="train_a3.2 — 2v2 PFSP, dong doi = agent dang train")
    p.add_argument("--initial-model", default="models/2v2 oriented/a3_1_checkpoints/snapshot_2750000.zip")
    p.add_argument("--resume",        default=None)
    p.add_argument("--steps",         default=100_000_000, type=int)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    os.makedirs("models/2v2 oriented/a3_2_checkpoints", exist_ok=True)

    pool = OpponentPoolA3_2()

    # Load a2.1_1 snapshots
    a3_1_snaps = sorted(
        glob.glob("models/2v2 oriented/a3_1_checkpoints/snapshot_*.zip"),
        key=lambda f: int(os.path.basename(f).replace("snapshot_", "").replace(".zip", ""))
    )
    for p in a3_1_snaps:
        pool.add(p)
    log.info(f"[A2.2_2] {len(a3_1_snaps)} a2.1_1 snapshot(s) loaded.")

    # Load a2.2_2 snapshots (if any exist from previous runs)
    a3_2_snaps = sorted(
        glob.glob("models/2v2 oriented/a3_2_checkpoints/snapshot_*.zip"),
        key=lambda f: int(os.path.basename(f).replace("snapshot_", "").replace(".zip", ""))
    )
    for p in a3_2_snaps:
        pool.add(p)
    if a3_2_snaps:
        log.info(f"[A2.2_2] {len(a3_2_snaps)} existing a2.2_2 snapshot(s) loaded.")

    log.info(f"[A2.2_2] Opponent pool khoi tao: {len(pool.pool_a3_1)} A2.1_1 agents, {len(pool.pool_a3_2)} A2.2_2 agents.")
    pool.log_stats()

    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    def make_env_1v1_1(): return SelfPlayEnvA3_2(opponent_pool=pool, fixed_mode='1v1_1')
    def make_env_dynamic(): return SelfPlayEnvA3_2(opponent_pool=pool, fixed_mode='dynamic_2v')

    # Fixed configuration to ensure stable vec_env agent sizes:
    # 1 x 1v1 (static 1 obs). The other 7 envs use dynamic_2v (switching between 2v1 and 2v2, but always 2 obs)
    env_fns = [
        make_env_1v1_1,
        make_env_dynamic,
        make_env_dynamic,
        make_env_dynamic,
        make_env_dynamic,
        make_env_dynamic,
        make_env_dynamic,
        make_env_dynamic,
    ]

    vec_env = VecMonitor(DummyVecEnv(env_fns))

    custom_objects = {k: v for k, v in PPO_PARAMS.items() if k != "policy_kwargs"}

    if args.resume:
        log.info(f"[A2.2_2] Resuming tu {args.resume}")
        model = PPO.load(args.resume, env=vec_env, custom_objects=custom_objects)
        model.set_env(vec_env)
        is_resume = True
    else:
        log.info(f"[A2.2_2] Tai model tu {args.initial_model}")
        model = PPO.load(args.initial_model, env=vec_env, custom_objects=custom_objects)
        model.set_env(vec_env)
        is_resume = False

    callback = SelfPlayCallbackA3_2(
        opponent_pool=pool,
        model_dir="models/2v2 oriented/",
        envs=vec_env,
        verbose=1
    )

    if is_resume:
        remaining_steps = max(args.steps - model.num_timesteps, 0)
        log.info(f"[A2.2_2] Resume tu step {model.num_timesteps:,} — con {remaining_steps:,} steps")
    else:
        remaining_steps = args.steps

    log.info(
        f"[A2.2_2] Bat dau — {len(env_fns)} envs thuc te ({vec_env.num_envs} agents/step) | {remaining_steps:,} steps | "
        f"{len(pool.pool_a3_1)} A2.1_1 + {len(pool.pool_a3_2)} A2.2_2 opponents | PFSP min_games={PFSP_MIN_GAMES}"
    )

    model.learn(
        total_timesteps     = remaining_steps,
        callback            = callback,
        reset_num_timesteps = not is_resume,
        log_interval        = 250_000 // (PPO_PARAMS["n_steps"] * N_ENVS),
        progress_bar        = True,
    )

    model.save("models/2v2 oriented/a3_2_final")
    log.info("[A2.2_2] Done -> models/2v2 oriented/a3_2_final.zip")
