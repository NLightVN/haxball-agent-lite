import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import math
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from training.env import HaxballCurriculumEnv, Disc, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP
from utils.model_logger import get_model_logger

log = get_model_logger("a1_fix")

# ── Config ─────────────────────────────────────────────────────────────────────
N_ENVS          = 8

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

# ── Self-Play Opponent Manager ────────────────────────────────────────────────
class OpponentManager:
    def __init__(self, initial_model_path):
        self.opponents = {}  # name -> policy
        self.old_pool = []
        self.new_pool = []
        self.terminated_pool = []
        
        self.old_idx = 0
        self.new_idx = 0
        
        # Stats per opponent
        self.stats = {} # name -> {'wins': 0, 'total': 0}

    def add_opponent(self, name, model_path, is_new=False):
        log.info(f"[SelfPlay] Loading opponent '{name}' from {model_path} into memory (is_new={is_new})...")
        policy = PPO.load(model_path, device='cpu')
        self.opponents[name] = policy
        self.stats[name] = {'wins': 0, 'total': 0}
        if is_new:
            self.new_pool.append(name)
        else:
            self.old_pool.append(name)
            
    def update_result(self, name, agent_won):
        if name in self.stats:
            self.stats[name]['total'] += 1
            if not agent_won:
                self.stats[name]['wins'] += 1
                
            h = self.stats[name]
            # Chỉ loại bỏ khỏi pool cũ, pool mới cứ giữ nguyên
            if name in self.old_pool and h['total'] >= 20:
                snapshot_winrate = h['wins'] / h['total']
                if snapshot_winrate < 0.05:
                    log.info(f"[SelfPlay] Opponent '{name}' winrate {snapshot_winrate:.3f} < 0.05 after {h['total']} matches. Moving to terminated pool.")
                    self.old_pool.remove(name)
                    self.terminated_pool.append(name)
                    self.old_idx = 0
            elif name in self.new_pool and h['total'] >= 20:
                snapshot_winrate = h['wins'] / h['total']
                if snapshot_winrate < 0.05:
                    log.info(f"[SelfPlay] Opponent '{name}' (new) winrate {snapshot_winrate:.3f} < 0.05. Moving to terminated pool.")
                    self.new_pool.remove(name)
                    self.terminated_pool.append(name)
                    self.new_idx = 0
        
    def sample_opponent(self):
        has_old = len(self.old_pool) > 0
        has_new = len(self.new_pool) > 0
        
        if not has_old and not has_new:
            return None, 'Static'

        use_new = False
        if has_old and has_new:
            use_new = np.random.rand() < 0.5
        elif has_new:
            use_new = True

        if use_new:
            idx = self.new_idx % len(self.new_pool)
            self.new_idx = (self.new_idx + 1) % len(self.new_pool)
            chosen_name = self.new_pool[idx]
        else:
            idx = self.old_idx % len(self.old_pool)
            self.old_idx = (self.old_idx + 1) % len(self.old_pool)
            chosen_name = self.old_pool[idx]
            
        return self.opponents[chosen_name], chosen_name

# ── Self-Play Environment Wrapper ─────────────────────────────────────────────
class FixA1Env(HaxballCurriculumEnv):
    def __init__(self, opponent_manager, **kwargs):
        super().__init__(**kwargs)
        self.opponent_manager = opponent_manager
        self.current_opponent_name = None
        
    def _reset_positions(self):
        # Call base to set up dimensions
        super()._reset_positions()
        
        # Override opponent
        if self.episode_type == 'opponent':
            policy, name = self.opponent_manager.sample_opponent()
            self.current_opponent_name = name
            if name in ('Wanderer', 'Pazzo', 'Random', 'Static'):
                self.opponent_type = name
                self.opponent_policy = None
            else:
                self.opponent_type = 'Trained'
                self.opponent_policy = policy

        # 100% Own goal situation
        own_goal_x = -self.HW * self._attack_sign
        own_goal_y = float(self._rng.uniform(-self.goal_y, self.goal_y))
        
        # Mọi player được spawn ngẫu nhiên trong đúng nửa sân của mình
        if self.team_id == 1:
            # Agent is RED (Left half), Opp is BLUE (Right half)
            ax = float(self._rng.uniform(-self.HW * 0.8 + PLYR_R, -PLYR_R))
            ox = float(self._rng.uniform(PLYR_R, self.HW * 0.8 - PLYR_R))
        else:
            # Agent is BLUE (Right half), Opp is RED (Left half)
            ax = float(self._rng.uniform(PLYR_R, self.HW * 0.8 - PLYR_R))
            ox = float(self._rng.uniform(-self.HW * 0.8 + PLYR_R, -PLYR_R))
            
        ay = float(self._rng.uniform(-self.goal_y, self.goal_y))
        oy = float(self._rng.uniform(-self.goal_y, self.goal_y))
        
        dx = own_goal_x - ax
        dy = own_goal_y - ay
        dist = math.hypot(dx, dy)
        
        if dist > 0:
            ndx, ndy = dx / dist, dy / dist
        else:
            ndx, ndy = 1.0, 0.0
            
        gap = float(self._rng.uniform(0.5, 2.0))
        b_dist = PLYR_R + BALL_R + gap
        bx = ax + ndx * b_dist
        by = ay + ndy * b_dist
        
        self.ball = Disc(bx, by, 0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)
        
        if self.team_id == 1:
            self.agents = [
                Disc(ax, ay, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP),
                Disc(ox, oy, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)
            ]
        else:
            self.agents = [
                Disc(ax, ay, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP),
                Disc(ox, oy, 0.0, 0.0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)
            ]
            
    def step(self, action):
        opponent_name = self.current_opponent_name
        obs, reward, terminated, truncated, info = super().step(action)
        
        if terminated or truncated:
            agent_score = self.scores[self.team_id - 1]
            opp_score = self.scores[2 - self.team_id]
            
            # Reduce goal reward from 30 to 10
            if agent_score > 0:
                reward -= 20.0
            
            if self.episode_type == 'opponent' and opponent_name is not None:
                if agent_score != opp_score:
                    agent_won = (agent_score > opp_score)
                    self.opponent_manager.update_result(opponent_name, agent_won)
                    
        return obs, reward, terminated, truncated, info

# ── Combined Callback ──────────────────────────────────────────────────────────
class SelfPlayCallback(BaseCallback):
    def __init__(self, opponent_manager, model_dir, envs, verbose=1):
        super().__init__(verbose)
        self.opponent_manager = opponent_manager
        self.model_dir = model_dir
        self.envs = envs
        self.snapshot_schedule = [1_000_000, 2_000_000, 4_000_000, 6_000_000]
        self.best_reward = -np.inf
        self.best_path = os.path.join(model_dir, "a1_fix_best")
        
    def _on_training_start(self) -> None:
        self.snapshot_schedule = [s for s in self.snapshot_schedule if s > self.num_timesteps]
        log.info(f"[SelfPlay] Remaining snapshot schedule: {self.snapshot_schedule}")
        
    def _on_step(self) -> bool:
        for env in self.envs.envs:
            env.total_timesteps_elapsed = self.num_timesteps
            
        if self.snapshot_schedule and self.num_timesteps >= self.snapshot_schedule[0]:
            target_step = self.snapshot_schedule.pop(0)
            snapshot_name = f"snapshot_{target_step}"
            path = os.path.join(self.model_dir, "a1_fix_checkpoints", f"{snapshot_name}.zip")
            self.model.save(path)
            
            if self.verbose:
                log.info(f"[SelfPlay] Saved snapshot to {path}")
            
            self.opponent_manager.add_opponent(snapshot_name, path, is_new=True)
            
            log.info("\n--- SelfPlay Opponent Winrates ---")
            log.info("Old Pool: " + ", ".join(self.opponent_manager.old_pool))
            log.info("New Pool: " + ", ".join(self.opponent_manager.new_pool))
            log.info("Terminated Pool: " + ", ".join(self.opponent_manager.terminated_pool))
            for name, h in self.opponent_manager.stats.items():
                if h['total'] > 0:
                    wr = h['wins'] / h['total']
                    log.info(f"  {name}: {wr*100:.1f}% ({h['wins']}/{h['total']})")
            log.info("----------------------------------")

        rew = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
        if rew is not None:
            if rew > self.best_reward:
                self.best_reward = rew
                self.model.save(self.best_path)

        return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--a1-model", default="models/a1_checkpoints/snapshot_21000000.zip", help="Path to initial A1 model")
    p.add_argument("--steps",    default=30_000_000,  type=int)
    p.add_argument("--resume",   default=None,        help="Resume from a previous checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Thêm patch để load được model train trên NumPy 2.x trong môi trường NumPy 1.x
    try:
        import numpy.core.numeric
        sys.modules['numpy._core'] = sys.modules['numpy.core']
        sys.modules['numpy._core.numeric'] = sys.modules['numpy.core.numeric']
        sys.modules['numpy._core.multiarray'] = sys.modules.get('numpy.core.multiarray', sys.modules['numpy.core'])
    except Exception:
        pass

    os.makedirs("models/a1_fix_checkpoints", exist_ok=True)

    opponent_manager = OpponentManager(args.a1_model)

    def make_env():
        return FixA1Env(opponent_manager=opponent_manager, phase='A1')

    vec_env = DummyVecEnv([make_env for _ in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    if args.resume:
        log.info(f"[SelfPlay] Resuming from {args.resume}")
        custom_objects = PPO_PARAMS.copy()
        if "policy_kwargs" in custom_objects:
            del custom_objects["policy_kwargs"]
        model = PPO.load(args.resume, env=vec_env, custom_objects=custom_objects)
        model.set_env(vec_env)
    else:
        log.info(f"[SelfPlay] Loading initial weights from {args.a1_model}")
        custom_objects = PPO_PARAMS.copy()
        if "policy_kwargs" in custom_objects:
            del custom_objects["policy_kwargs"]
        model = PPO.load(args.a1_model, env=vec_env, custom_objects=custom_objects)
        model.set_env(vec_env)

    is_resume = bool(args.resume)

    # Tải toàn bộ pool cũ
    old_ckpt_dir = "models/a1_checkpoints"
    if os.path.exists(old_ckpt_dir):
        existing_old = sorted(
            [f for f in os.listdir(old_ckpt_dir) if f.startswith("snapshot_") and f.endswith(".zip")],
            key=lambda f: int(f.replace("snapshot_", "").replace(".zip", ""))
        )
        for fname in existing_old:
            snap_name = fname.replace(".zip", "")
            snap_path = os.path.join(old_ckpt_dir, fname)
            opponent_manager.add_opponent(snap_name, snap_path, is_new=False)
            
    if len(opponent_manager.old_pool) == 0:
        opponent_manager.add_opponent("snapshot_21000000", args.a1_model, is_new=False)

    # Tải pool mới nếu có (resume)
    ckpt_dir = "models/a1_fix_checkpoints"
    if os.path.exists(ckpt_dir):
        existing_new = sorted(
            [f for f in os.listdir(ckpt_dir) if f.startswith("snapshot_") and f.endswith(".zip")],
            key=lambda f: int(f.replace("snapshot_", "").replace(".zip", ""))
        )
        for fname in existing_new:
            snap_name = fname.replace(".zip", "")
            snap_path = os.path.join(ckpt_dir, fname)
            opponent_manager.add_opponent(snap_name, snap_path, is_new=True)

    callback = SelfPlayCallback(
        opponent_manager=opponent_manager,
        model_dir="models/",
        envs=vec_env,
        verbose=1
    )

    remaining_steps = max(args.steps - model.num_timesteps, 0) if is_resume else args.steps

    log.info(f"[SelfPlay] Starting — {N_ENVS} envs, {remaining_steps:,} steps to go")

    model.learn(
        total_timesteps      = remaining_steps,
        callback             = callback,
        reset_num_timesteps  = not is_resume,
        log_interval         = 250_000 // (PPO_PARAMS["n_steps"] * N_ENVS),
        progress_bar         = True,
    )

    model.save("models/a1_fix_final")
    log.info("[SelfPlay] Done -> models/a1_fix_final.zip")
