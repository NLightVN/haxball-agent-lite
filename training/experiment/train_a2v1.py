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

log = get_model_logger("a2_train")

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
    verbose      = 1
)

# ── Self-Play Opponent Manager ────────────────────────────────────────────────
class OpponentManager:
    def __init__(self, initial_model_path, dynamic_prob=False):
        self.opponents = {}  # name -> policy
        self.old_pool = []
        self.new_pool = []
        self.terminated_pool = []
        self.dynamic_prob = dynamic_prob
        
        self.old_idx = 0
        self.new_idx = 0
        self.stats = {} 

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
            if name in self.old_pool and h['total'] >= 20:
                snapshot_winrate = h['wins'] / h['total']
                if snapshot_winrate < 0.05:
                    self.old_pool.remove(name)
                    self.terminated_pool.append(name)
                    self.old_idx = 0
            elif name in self.new_pool and h['total'] >= 20:
                snapshot_winrate = h['wins'] / h['total']
                if snapshot_winrate < 0.05:
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
            if self.dynamic_prob:
                num_new = len(self.new_pool)
                # max(60%, min(num_new * 10%, 100%))
                prob = max(0.6, min(num_new * 0.1, 1.0))
                use_new = np.random.rand() < prob
            else:
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
class A2Env(HaxballCurriculumEnv):
    def __init__(self, opponent_manager, teammate_policy, **kwargs):
        super().__init__(**kwargs)
        self.opponent_manager = opponent_manager
        self.teammate_policy = teammate_policy
        self.current_opponent_name = None
        
    def _reset_positions(self):
        super()._reset_positions()
        
        if self.episode_type == 'opponent':
            policy, name = self.opponent_manager.sample_opponent()
            self.current_opponent_name = name
            if name in ('Wanderer', 'Pazzo', 'Random', 'Static'):
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
            opp_score = self.scores[2 - self.team_id]
            
            if self.episode_type == 'opponent' and opponent_name is not None:
                if agent_score != opp_score:
                    agent_won = (agent_score > opp_score)
                    self.opponent_manager.update_result(opponent_name, agent_won)
                    
        return obs, reward, terminated, truncated, info


def get_next_snapshot_step(current_step, start_step=1_000_000, factor=1.5):
    step = start_step
    while step <= current_step:
        step = int(step * factor)
    return step

class A2Callback(BaseCallback):
    def __init__(self, opponent_manager_self, opponent_manager_opp, team_name, model_dir, envs, verbose=1):
        super().__init__(verbose)
        self.opponent_manager_self = opponent_manager_self
        self.opponent_manager_opp = opponent_manager_opp
        self.team_name = team_name
        self.model_dir = model_dir
        self.envs = envs
        self.best_reward = -np.inf
        self.best_path = os.path.join(model_dir, f"a2_{team_name}_best")
        
    def _on_step(self) -> bool:
        for env in self.envs.envs:
            env.total_timesteps_elapsed = self.num_timesteps
            
        next_snap = get_next_snapshot_step(self.num_timesteps - 1)
        if self.num_timesteps >= next_snap:
            snapshot_name = f"snapshot_{next_snap}"
            path = os.path.join(self.model_dir, f"a2_{self.team_name}_checkpoints", f"{snapshot_name}.zip")
            self.model.save(path)
            
            if self.verbose:
                log.info(f"[{self.team_name}] Saved snapshot to {path}")
            
            # The opponent manager for the OTHER team receives this checkpoint
            self.opponent_manager_opp.add_opponent(snapshot_name, path, is_new=True)
            
        rew = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
        if rew is not None and rew > self.best_reward:
            self.best_reward = rew
            self.model.save(self.best_path)
            
        if self.num_timesteps % 50_000 == 0:
            if self.verbose:
                opp_mgr = self.opponent_manager_opp
                log.info(f"[{self.team_name}] Step {self.num_timesteps} | Opponent Pool (Old: {len(opp_mgr.old_pool)}, New: {len(opp_mgr.new_pool)})")
                for name, h in list(opp_mgr.stats.items())[-5:]: # Chỉ in 5 cái gần nhất cho đỡ rối
                    if h['total'] > 0:
                        wr = h['wins'] / h['total']
                        log.info(f"  - Opponent {name}: Winrate = {wr:.1%} ({h['wins']}/{h['total']})")

        return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="models/a2_base.zip", help="Path to initial expanded model")
    p.add_argument("--steps",    default=30_000_000,  type=int)
    p.add_argument("--chunk-steps", default=250_000, type=int, help="Steps per alternating training chunk")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load patch for numpy
    try:
        import numpy.core.numeric
        sys.modules['numpy._core'] = sys.modules['numpy.core']
        sys.modules['numpy._core.numeric'] = sys.modules['numpy.core.numeric']
        sys.modules['numpy._core.multiarray'] = sys.modules.get('numpy.core.multiarray', sys.modules['numpy.core'])
    except Exception:
        pass

    os.makedirs("models/a2_t1_checkpoints", exist_ok=True) # Team 1 (2 players)
    os.makedirs("models/a2_t2_checkpoints", exist_ok=True) # Team 2 (1 player)

    # Opponent Managers (Pools)
    # T1 manager manages T1's checkpoints, used as opponents when training T2
    t1_pool = OpponentManager(args.base_model, dynamic_prob=False)
    t1_pool.add_opponent("base", args.base_model, is_new=False)
    
    # T2 manager manages T2's checkpoints, used as opponents when training T1
    # For T1 (2-player team), use dynamic probability for its opponent pool (t2_pool).
    t2_pool = OpponentManager(args.base_model, dynamic_prob=True)
    t2_pool.add_opponent("base", args.base_model, is_new=False)

    # Load legacy A1 checkpoints into t2_pool as old pool
    for a1_dir in ["models/a1_checkpoints", "models/a1_fix_checkpoints"]:
        if os.path.exists(a1_dir):
            existing_a1 = sorted(
                [f for f in os.listdir(a1_dir) if f.startswith("snapshot_") and f.endswith(".zip")],
                key=lambda f: int(f.replace("snapshot_", "").replace(".zip", ""))
            )
            for fname in existing_a1:
                snap_name = f"{os.path.basename(a1_dir)}_{fname.replace('.zip', '')}"
                snap_path = os.path.join(a1_dir, fname)
                t2_pool.add_opponent(snap_name, snap_path, is_new=False)

    def make_env_t1():
        # Env for training T1 (n_agents=2). Uses T2's pool for opponents.
        return A2Env(opponent_manager=t2_pool, teammate_policy=None, phase='A2.0', n_agents=2)
        
    def make_env_t2():
        # Env for training T2 (n_agents=1). Uses T1's pool for opponents. Teammate policy is None because n_agents=1.
        return A2Env(opponent_manager=t1_pool, teammate_policy=None, phase='A2.0', n_agents=1)

    vec_env_t1 = DummyVecEnv([make_env_t1 for _ in range(N_ENVS)])
    vec_env_t1 = VecMonitor(vec_env_t1)
    
    vec_env_t2 = DummyVecEnv([make_env_t2 for _ in range(N_ENVS)])
    vec_env_t2 = VecMonitor(vec_env_t2)

    log.info(f"Loading initial weights from {args.base_model}")
    model_t1 = PPO.load(args.base_model, env=vec_env_t1, custom_objects=PPO_PARAMS)
    model_t1.set_env(vec_env_t1)
    
    model_t2 = PPO.load(args.base_model, env=vec_env_t2, custom_objects=PPO_PARAMS)
    model_t2.set_env(vec_env_t2)
    
    # Load existing pools if continuing
    for pool, ckpt_dir in [(t1_pool, "models/a2_t1_checkpoints"), (t2_pool, "models/a2_t2_checkpoints")]:
        if os.path.exists(ckpt_dir):
            existing = sorted(
                [f for f in os.listdir(ckpt_dir) if f.startswith("snapshot_") and f.endswith(".zip")],
                key=lambda f: int(f.replace("snapshot_", "").replace(".zip", ""))
            )
            for fname in existing:
                snap_name = fname.replace(".zip", "")
                snap_path = os.path.join(ckpt_dir, fname)
                pool.add_opponent(snap_name, snap_path, is_new=True)

    # Set teammate policy for T1 envs dynamically so it uses the current model_t1 weights
    for env in vec_env_t1.envs:
        env.teammate_policy = model_t1

    cb_t1 = A2Callback(t1_pool, t2_pool, "t1", "models/", vec_env_t1)
    cb_t2 = A2Callback(t2_pool, t1_pool, "t2", "models/", vec_env_t2)

    total_steps = args.steps
    chunk = args.chunk_steps
    
    log.info(f"Starting Alternating Training: {total_steps:,} steps total. Chunk size: {chunk:,}")
    
    # Simple loop
    while model_t1.num_timesteps < total_steps or model_t2.num_timesteps < total_steps:
        # Train T1 (Team 1, 2 players)
        log.info(f"\n--- Training T1 (Step {model_t1.num_timesteps:,} / {total_steps:,}) ---")
        # Ensure teammate policy is updated (it points to model_t1 object, but good to ensure)
        for env in vec_env_t1.envs:
            env.teammate_policy = model_t1
        
        model_t1.learn(
            total_timesteps=chunk,
            callback=cb_t1,
            reset_num_timesteps=False,
            log_interval=10,
            progress_bar=True,
        )
        model_t1.save("models/a2_t1_final")
        
        # Train T2 (Team 2, 1 player)
        log.info(f"\n--- Training T2 (Step {model_t2.num_timesteps:,} / {total_steps:,}) ---")
        model_t2.learn(
            total_timesteps=chunk,
            callback=cb_t2,
            reset_num_timesteps=False,
            log_interval=10,
            progress_bar=True,
        )
        model_t2.save("models/a2_t2_final")

    log.info("Finished Alternating Training!")
