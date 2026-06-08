"""
train_a3.3.py — 3v3 Self-Play Training with PFSP.

Cau truc:
  - 20% matches: 1v1 (agent vs 1 opponent tu pool).
  - 80% matches: 3v3 (agent + clone hien tai) vs (1 opponent tu pool).

Pool duy nhat: opponent pool gom tat ca a1.1_1 + a1.1_1 + a2.3_3 snapshots.
Dong doi = chinh agent dang duoc train (current_model).

PFSP (Prioritized Fictitious Self-Play):
  Trong hai opponent co winrate gan 0.5 nhat, vi do la nhung match kho nhat
  va huu ich nhat cho training. Opponent co winrate > 0.95 (qua de) hoac < 0.05
  (qua kho) se it duoc chon hon nhung van duoc giu lai trong pool.
  weight(wr) = max(wr * (1 - wr) * 4, floor=0.05)  — dinh tai wr=0.5 (=1.0)
"""

import argparse
import glob
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

log = get_model_logger("a2.3_3")

# ── Config ─────────────────────────────────────────────────────────────────────
N_ENVS            = 8
TARGET_REWARD     = 15.0
SNAPSHOT_INTERVAL = 500_000

# PFSP params
PFSP_MIN_GAMES   = 20     # so tran toi thieu truoc khi ap dung PFSP
PFSP_FLOOR       = 0.05   # trong so toi thieu cho moi opponent

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
class OpponentPoolA3:
    """
    Pool opponent voi PFSP sampling.

    Winrate = ti le thang cua AGENT (wr cao = agent manh hon opponent).
    PFSP uu tien opponent co wr ~50%: bat buoc agent phai hoc nhung tran kho.

    Convention (nhat quan voi train_a1.1.py):
      stats[path]['wins']  = so lan SNAPSHOT (opponent) thang (agent thua)
      snapshot_winrate     = wins / total
      agent_winrate        = 1 - snapshot_winrate

    PFSP weight function: w(agent_wr) = max(4 * agent_wr * (1 - agent_wr), PFSP_FLOOR)
      Dinh tai agent_wr = 0.5 (ngang suc) = 1.0, san tai 0.0/1.0 = floor.
      wr=0.5 -> w=1.00 (vua suc, chon nhieu nhat)
      wr=1.0 -> w=0.05 (qua de,  it chon)
    """
    def __init__(self, min_games: int = PFSP_MIN_GAMES, floor: float = PFSP_FLOOR):
        self.pool: list[str] = []         # danh sach duong dan model (tat ca)
        self.pool_1v1: list[str] = []     # danh sach chi gom cac model 1v1
        self.pool_3v3: list[str] = []     # danh sach chi gom cac model 3v3
        self._loaded: dict[str, PPO] = {} # cache: path -> policy
        self.stats: dict[str, dict] = {}  # path -> {'wins':int, 'total':int}
        self.min_games = min_games
        self.floor = floor

    # ── Pool management ───────────────────────────────────────────────────────
    def add(self, path: str, is_1v1_model: bool = False):
        if path not in self.pool:
            self.pool.append(path)
            if is_1v1_model:
                self.pool_1v1.append(path)
            else:
                self.pool_3v3.append(path)
            self.stats[path] = {'wins': 0, 'total': 0}  # wins = snapshot (opponent) wins
            log.info(f"[Pool] + {os.path.basename(path)}  (total={len(self.pool)}, 1v1={len(self.pool_1v1)}, 3v3={len(self.pool_3v3)})")

    def get_policy(self, path: str) -> PPO:
        if path not in self._loaded:
            log.info(f"[Pool] Loading {path}...")
            self._loaded[path] = PPO.load(path, device='cpu')
        return self._loaded[path]

    # ── Result tracking ───────────────────────────────────────────────────────
    def update_result(self, path: str, agent_won: bool):
        """Cap nhat ket qua sau moi tran (chi khi co nguoi thang ro rang).
        wins = so lan SNAPSHOT thang (nhat quan voi train_a1.1.py).
        """
        if path not in self.stats:
            return
        s = self.stats[path]
        s['total'] += 1
        if not agent_won:          # snapshot thang <=> agent thua
            s['wins'] += 1

    # ── PFSP weight ───────────────────────────────────────────────────────────
    def _weight(self, path: str) -> float:
        s = self.stats.get(path, {'wins': 0, 'total': 0})
        if s['total'] < self.min_games:
            return 1.0  # uniform truoc khi du data
        snapshot_wr = s['wins'] / s['total']   # opponent winrate
        agent_wr    = 1.0 - snapshot_wr        # agent winrate
        return max(4.0 * agent_wr * (1.0 - agent_wr), self.floor)

    # ── Sampling ──────────────────────────────────────────────────────────────
    def sample(self, is_1v1: bool = False, is_3v3_only: bool = False) -> tuple[PPO | None, str | None]:
        """PFSP sampling: tra ve (policy, path). Neu is_3v3_only=True, sample tu pool_3v3."""
        if is_1v1:
            target_pool = self.pool_1v1
        elif is_3v3_only:
            target_pool = self.pool_3v3 if self.pool_3v3 else self.pool
        else:
            target_pool = self.pool
            
        if not target_pool:
            return None, None
            
        weights = np.array([self._weight(p) for p in target_pool], dtype=np.float64)
        weights /= weights.sum()
        path = np.random.choice(target_pool, p=weights)
        return self.get_policy(path), path

    def sample_uniform(self) -> tuple[PPO | None, str | None]:
        """Uniform sampling tu toan bo pool (bo qua PFSP)."""
        if not self.pool:
            return None, None
        path = np.random.choice(self.pool)
        return self.get_policy(path), path

    # ── Stats summary ─────────────────────────────────────────────────────────
    def log_stats(self):
        log.info(f"\n{'-'*60}")
        log.info(f"  PFSP Opponent Pool — {len(self.pool)} agents")
        log.info(f"  (snapshot_wr = opponent wins / total | agent_wr = 1 - snapshot_wr)")
        log.info(f"{'-'*60}")
        weights = {p: self._weight(p) for p in self.pool}
        total_w = sum(weights.values()) or 1.0
        for path in self.pool:
            s = self.stats[path]
            name = os.path.basename(path).replace('.zip', '')
            w = weights[path] / total_w * 100
            if s['total'] >= self.min_games:
                snap_wr  = s['wins'] / s['total']    # opponent winrate
                agent_wr = 1.0 - snap_wr
                log.info(
                    f"  {name:<40} "
                    f"opp_wr={snap_wr*100:5.1f}%  agent_wr={agent_wr*100:5.1f}%  "
                    f"({s['wins']:>4}/{s['total']:<4})  sel={w:.1f}%"
                )
            else:
                log.info(
                    f"  {name:<40} "
                    f"opp_wr=  ?      agent_wr=  ?     "
                    f"({s['wins']:>4}/{s['total']:<4}  <{self.min_games})  sel={w:.1f}%"
                )
        log.info(f"{'-'*60}")


# ── Self-Play Environment ─────────────────────────────────────────────────────
class SelfPlayEnvA3(HaxballCurriculumEnv):
    """
    - phase='A2.3_3': dung map_3v3.json.
    - p_1v1: xac suat 1v1 (mac dinh 0.2).
    - Dong doi: current_model (agent dang train).
    - Doi thu: PFSP sampling tu OpponentPoolA3.
    - Track ket qua tung tran de cap nhat winrate.
    """
    def __init__(self, opponent_pool: OpponentPoolA3, p_1v1: float = 0.2, fixed_mode: str = None, **kwargs):
        super().__init__(phase='A2.3_3', p_1v1=p_1v1, fixed_mode=fixed_mode, **kwargs)
        self.opponent_pool = opponent_pool
        self._current_opp_paths: list[str] = []   # luu paths de bao cao ket qua

    def _reset_positions(self):
        super()._reset_positions()

        if self.episode_type == 'opponent':
            self.opponent_type = 'Trained'
            self.teammate_policy = None  # dung current_model lam dong doi
            
            if getattr(self, 'is_1v1', False):
                # Tran 1v1: chi chon doi thu 1v1
                policy, path = self.opponent_pool.sample(is_1v1=True)
                self._current_opp_paths = []  # winrate chi tinh trong 3v3 dong nhat
                self.opponent_policies = [policy] if policy else []
            elif getattr(self, 'is_3v3_chaos', False):
                # Tran 3v3 chaos: boc 3 doi thu ngau nhien tu toan bo pool
                pol1, path1 = self.opponent_pool.sample_uniform()
                pol2, path2 = self.opponent_pool.sample_uniform()
                pol3, path3 = self.opponent_pool.sample_uniform()
                self._current_opp_paths = []  # winrate chi tinh trong 3v3 dong nhat
                self.opponent_policies = [p for p in (pol1, pol2, pol3) if p]
            else:
                # 3v3 PFSP dong nhat: 1 doi thu nhan 3 tu pool 3v3
                pol1, path1 = self.opponent_pool.sample(is_3v3_only=True)
                self._current_opp_paths = [path1] if path1 else []
                self.opponent_policies = [pol1, pol1, pol1] if pol1 else []

    def step(self, action):
        # Luu paths truoc khi super().step() co the goi _reset_positions()
        opp_paths = self._current_opp_paths

        obs, reward, terminated, truncated, info = super().step(action)

        # Bao cao ket qua khi tran ket thuc
        if (terminated or truncated) and opp_paths:
            agent_score = self.scores[self.team_id - 1]
            opp_score   = self.scores[2 - self.team_id]
            if agent_score != opp_score:  # bo qua hoa
                agent_won = agent_score > opp_score
                for p in opp_paths:
                    self.opponent_pool.update_result(p, agent_won)

        return obs, reward, terminated, truncated, info


# ── Callback ───────────────────────────────────────────────────────────────────
class SelfPlayCallback(BaseCallback):
    def __init__(self, opponent_pool: OpponentPoolA3, model_dir: str, envs, verbose=1):
        super().__init__(verbose)
        self.opponent_pool = opponent_pool
        self.model_dir = model_dir
        self.envs = envs
        self.ckpt_dir = os.path.join(model_dir, "a3_3_checkpoints")
        self.best_reward = -np.inf
        self.best_path = os.path.join(model_dir, "a3_3_best")
        self.next_snapshot_at = 3_000_000
        self.last_log_step = 0

    def _get_next_snapshot_step(self, N: int) -> int:
        accumulated = sum((k + 3) * 1_000_000 for k in range(N))
        return accumulated + (N + 3) * 1_000_000

    def _on_training_start(self) -> None:
        existing = [f for f in os.listdir(self.ckpt_dir)
                    if f.startswith("snapshot_") and f.endswith(".zip")]
        N = len(existing)
        self.next_snapshot_at = self._get_next_snapshot_step(N)
        if N > 0:
            log.info(f"[A2.3_3] Resume: {N} snapshot(s) da co, next snapshot tai step {self.next_snapshot_at:,}")
        else:
            log.info(f"[A2.3_3] Next snapshot tai step {self.next_snapshot_at:,}")

    def _on_step(self) -> bool:
        # 1. Dong bo timestep va current_model (dong doi = agent dang train)
        for env in self.envs.envs:
            env.total_timesteps_elapsed = self.num_timesteps
            env.current_model = self.model

        # 2. Log PFSP stats dinh ky
        if self.num_timesteps - self.last_log_step >= 250_000:
            self.opponent_pool.log_stats()
            self.last_log_step = self.num_timesteps

        # 3. Luu snapshot va them vao opponent pool
        while self.next_snapshot_at <= self.num_timesteps:
            snap_name = f"snapshot_{self.num_timesteps}"
            path = os.path.join(self.ckpt_dir, f"{snap_name}.zip")
            self.model.save(path)
            log.info(f"[A2.3_3] Saved snapshot -> {path}")

            # Them vao pool doi thu (a2.3_3 snapshot = 3v3 model, khong dung cho 1v1)
            self.opponent_pool.add(path, is_1v1_model=False)

            existing = [f for f in os.listdir(self.ckpt_dir)
                        if f.startswith("snapshot_") and f.endswith(".zip")]
            N = len(existing)
            self.next_snapshot_at = self._get_next_snapshot_step(N)

        # 4. Best reward tracking
        rew = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
        if rew is not None:
            if rew > self.best_reward:
                self.best_reward = rew
                self.model.save(self.best_path)
                if self.verbose:
                    log.info(f"[A2.3_3] New best: {rew:.3f} -> {self.best_path}.zip")
            if rew >= TARGET_REWARD:
                log.info(f"[A2.3_3] Target reached! ep_rew_mean={rew:.3f}")
                return False

        return True


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="train_a3.3 — 3v3 PFSP, dong doi = agent dang train")
    p.add_argument("--initial-model", default="models/1v1 oriented/a1_1_final.zip")
    p.add_argument("--resume",        default=None)
    p.add_argument("--steps",         default=100_000_000, type=int)
    p.add_argument("--p-1v1",         default=0.2,         type=float,
                   help="Xac suat episode 1v1 (mac dinh 0.2)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    os.makedirs("models/2v2 oriented/a3_3_checkpoints", exist_ok=True)

    # 1. Khoi tao opponent pool
    pool = OpponentPoolA3()

    # a1.1_1 snapshots — toan bo 1v1 agents da train
    a1_1_snaps = sorted(
        glob.glob("models/1v1 oriented/a1_1_checkpoints/snapshot_*.zip"),
        key=lambda f: int(os.path.basename(f).replace("snapshot_", "").replace(".zip", ""))
    )
    for p in a1_1_snaps:
        pool.add(p, is_1v1_model=True)
    log.info(f"[A2.3_3] {len(a1_1_snaps)} a1.1_1 snapshot(s) loaded.")

    if os.path.exists("models/1v1 oriented/a1_1_final.zip"):
        pool.add("models/1v1 oriented/a1_1_final.zip", is_1v1_model=True)
    else:
        log.warning("[A2.3_3] models/1v1 oriented/a1_1_final.zip not found!")

    # finetuned snapshots — a1.1_1 snapshots da duoc fine-tune sang map 3v3 (neu co)
    finetuned_snaps = sorted(
        glob.glob("models/1v1 oriented/a1_1_checkpoints/finetuned_snapshot_*.zip"),
        key=lambda f: int(os.path.basename(f).replace("finetuned_snapshot_", "").replace(".zip", ""))
    )
    for p in finetuned_snaps:
        pool.add(p, is_1v1_model=True)
    if finetuned_snaps:
        log.info(f"[A2.3_3] {len(finetuned_snaps)} finetuned snapshot(s) loaded.")

    # a2.3_3 snapshots hien co (khi resume)
    a3_3_snaps = sorted(
        glob.glob("models/2v2 oriented/a3_3_checkpoints/snapshot_*.zip"),
        key=lambda f: int(os.path.basename(f).replace("snapshot_", "").replace(".zip", ""))
    )
    for p in a3_3_snaps:
        pool.add(p, is_1v1_model=False)
    if a3_3_snaps:
        log.info(f"[A2.3_3] {len(a3_3_snaps)} existing a2.3_3 snapshot(s) loaded.")

    log.info(f"[A2.3_3] Opponent pool khoi tao: {len(pool.pool)} agents.")
    pool.log_stats()

    # 2. Tao moi truong
    from training.vec_env import HaxballMultiAgentVecEnv

    def make_env_1v1_1(): return SelfPlayEnvA3(opponent_pool=pool, p_1v1=args.p_1v1, fixed_mode='1v1_1')
    def make_env_1v1_2(): return SelfPlayEnvA3(opponent_pool=pool, p_1v1=args.p_1v1, fixed_mode='1v1_2')
    def make_env_3v3():   return SelfPlayEnvA3(opponent_pool=pool, p_1v1=args.p_1v1, fixed_mode='3v3')

    env_fns = [
        make_env_1v1_1,
        make_env_1v1_2,
        make_env_3v3,
        make_env_3v3,
        make_env_3v3,
        make_env_3v3,
        make_env_3v3,
        make_env_3v3,
    ]

    vec_env = HaxballMultiAgentVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    # 3. Tai model
    custom_objects = {k: v for k, v in PPO_PARAMS.items() if k != "policy_kwargs"}

    if args.resume:
        log.info(f"[A2.3_3] Resuming tu {args.resume}")
        model = PPO.load(args.resume, env=vec_env, custom_objects=custom_objects)
        model.set_env(vec_env)
        is_resume = True
    else:
        log.info(f"[A2.3_3] Tai model tu {args.initial_model}")
        model = PPO.load(args.initial_model, env=vec_env, custom_objects=custom_objects)
        model.set_env(vec_env)
        is_resume = False

    # 4. Callback
    callback = SelfPlayCallback(
        opponent_pool=pool,
        model_dir="models/2v2 oriented/",
        envs=vec_env,
        verbose=1
    )

    # 5. Steps con lai
    if is_resume:
        remaining_steps = max(args.steps - model.num_timesteps, 0)
        log.info(f"[A2.3_3] Resume tu step {model.num_timesteps:,} — con {remaining_steps:,} steps")
    else:
        remaining_steps = args.steps

    log.info(
        f"[A2.3_3] Bat dau — {len(env_fns)} envs thuc te ({vec_env.num_envs} agents/step) | {remaining_steps:,} steps | "
        f"p_1v1={args.p_1v1} | {len(pool.pool)} opponents | PFSP min_games={PFSP_MIN_GAMES}"
    )

    # 6. Train
    model.learn(
        total_timesteps     = remaining_steps,
        callback            = callback,
        reset_num_timesteps = not is_resume,
        log_interval        = 250_000 // (PPO_PARAMS["n_steps"] * N_ENVS),
        progress_bar        = True,
    )

    model.save("models/2v2 oriented/a3_3_final")
    log.info("[A2.3_3] Done -> models/2v2 oriented/a3_3_final.zip")
