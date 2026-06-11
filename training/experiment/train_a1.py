"""
train_a1.2.py — Self-Play Training.

Episode types:
  • PRECISION : small goal (0.3–0.6×), no opponent → aim accuracy
  • OPPONENT  : goal 1.4×→0.6× over 2M steps, opponent = Sampled from self-play pool

Usage:
    python training/train_a1.2.py
"""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.env import HaxballCurriculumEnv
from utils.model_logger import get_model_logger

log = get_model_logger("a1")

# ── Config ─────────────────────────────────────────────────────────────────────
N_ENVS          = 8
TARGET_REWARD   = 15.0
BOT_TYPES       = ('Wanderer', 'Pazzo', 'Random', 'Static')

PPO_PARAMS = dict(
    n_steps      = 2048,
    batch_size   = 512,
    n_epochs     = 10,
    gamma        = 0.99,
    gae_lambda   = 0.95,
    ent_coef     = 0.02,
    learning_rate= 2e-4,     # slightly lower for fine-tuning stability
    clip_range   = 0.2,
    vf_coef      = 0.5,
    max_grad_norm= 0.5,
    policy_kwargs= dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
)

# ── Self-Play Opponent Manager ────────────────────────────────────────────────
class OpponentManager:
    def __init__(self, initial_model_path):
        self.opponents = {}  # name -> policy
        self.agent_pool = []
        self.bot_pool = list(BOT_TYPES)
        self.terminated_pool = []
        
        self.agent_idx = 0
        self.terminated_idx = 0
        
        # Stats per opponent: wins = snapshot's wins (agent lost)
        self.stats = {} # name -> {'wins': 0, 'total': 0}

    def add_opponent(self, name, model_path):
        log.info(f"[SelfPlay] Loading opponent '{name}' from {model_path} into memory...")
        policy = PPO.load(model_path, device='cpu')
        self.opponents[name] = policy
        self.stats[name] = {'wins': 0, 'total': 0}
        self.agent_pool.append(name)
            
    def update_result(self, name, agent_won):
        if name in self.stats:
            self.stats[name]['total'] += 1
            # "winrate cua tung snapshot": if agent won, snapshot lost.
            if not agent_won:
                self.stats[name]['wins'] += 1
                
            # Check for termination if in active pool
            # Require at least 20 matches to have a statistically meaningful winrate
            h = self.stats[name]
            if name in self.agent_pool and h['total'] >= 20:
                snapshot_winrate = h['wins'] / h['total']
                if snapshot_winrate < 0.05:
                    log.info(f"[SelfPlay] Opponent '{name}' winrate {snapshot_winrate:.3f} < 0.05 after {h['total']} matches. Moving to terminated pool.")
                    if name in self.agent_pool:
                        self.agent_pool.remove(name)
                    self.terminated_pool.append(name)
                    # Reset indices to avoid out of bounds
                    self.agent_idx = 0
                    self.terminated_idx = 0
        
    def sample_opponent(self):
        num_agents = len(self.agent_pool)
        total_entities = num_agents + 1  # The +1 is the bot entity

        idx = self.agent_idx % total_entities
        self.agent_idx = (self.agent_idx + 1) % total_entities

        if idx == num_agents:
            # Bot's turn
            r = np.random.rand()
            if r < 0.82:
                bot_name = 'Wanderer'
            elif r < 0.88:
                bot_name = 'Pazzo'
            elif r < 0.94:
                bot_name = 'Random'
            else:
                bot_name = 'Static'
            return None, bot_name
        else:
            # Snapshot's turn
            chosen_name = self.agent_pool[idx]
            return self.opponents[chosen_name], chosen_name

# ── Self-Play Environment Wrapper ─────────────────────────────────────────────
class SelfPlayEnv(HaxballCurriculumEnv):
    def __init__(self, opponent_manager, **kwargs):
        super().__init__(**kwargs)
        self.opponent_manager = opponent_manager
        self.current_opponent_name = None
        
    def _reset_positions(self):
        # Call base to set up
        super()._reset_positions()
        
        # Override opponent if it's an opponent episode
        if self.episode_type == 'opponent':
            # Sample either a snapshot opponent or one of the four bot types
            policy, name = self.opponent_manager.sample_opponent()
            self.current_opponent_name = name
            if name in BOT_TYPES:
                self.opponent_type = name
                self.opponent_policy = None
            else:
                self.opponent_type = 'Trained'
                self.opponent_policy = policy
            
    def step(self, action):
        # Capture opponent for this episode before env.step() can reset and resample.
        opponent_name = self.current_opponent_name
        obs, reward, terminated, truncated, info = super().step(action)
        
        if terminated or truncated:
            # Determine match result
            # team_id is 1 (RED) or 2 (BLUE). scores array is [RED, BLUE]
            agent_score = self.scores[self.team_id - 1]
            opp_score = self.scores[2 - self.team_id]
            
            if self.episode_type == 'opponent' and opponent_name is not None:
                if agent_score != opp_score:
                    agent_won = (agent_score > opp_score)
                    self.opponent_manager.update_result(opponent_name, agent_won)
                    
        return obs, reward, terminated, truncated, info

# ── Combined Callback ──────────────────────────────────────────────────────────
class SelfPlayCallback(BaseCallback):
    def __init__(self, opponent_manager, target_reward, model_dir, envs, verbose=1):
        super().__init__(verbose)
        self.opponent_manager = opponent_manager
        self.target_reward = target_reward
        self.model_dir = model_dir
        self.envs = envs
        # next_snapshot_at will be recalculated in _on_training_start based on existing snapshots
        self.next_snapshot_at = 1_000_000
        self.best_reward = -np.inf
        self.best_path = os.path.join(model_dir, "a1_best")
        
    def _on_training_start(self) -> None:
        # Recalculate next_snapshot_at based on snapshots already loaded (resume case)
        N = len(self.opponent_manager.agent_pool)
        if N > 0:
            # Rebuild next_snapshot_at: sum of (k+1)*1_000_000 for k=0..N-1, then next interval
            accumulated = sum((k + 1) * 1_000_000 for k in range(N))
            next_interval = (N + 1) * 1_000_000
            self.next_snapshot_at = accumulated + next_interval
            log.info(f"[SelfPlay] Resume: {N} existing snapshots, next snapshot at step {self.next_snapshot_at:,}")
        else:
            self.next_snapshot_at = 1_000_000
        
    def _on_step(self) -> bool:
        # 1. Sync timesteps so env curriculum sees current progress
        for env in self.envs.envs:
            env.total_timesteps_elapsed = self.num_timesteps
            
        # 2. Checkpoint and register new opponent
        while self.next_snapshot_at <= self.num_timesteps:
            snapshot_name = f"snapshot_{self.num_timesteps}"
            path = os.path.join(self.model_dir, "a1_checkpoints", f"{snapshot_name}.zip")
            self.model.save(path)
            
            if self.verbose:
                log.info(f"[SelfPlay] Saved snapshot to {path}")
            
            self.opponent_manager.add_opponent(snapshot_name, path)
            
            # Dynamic interval: (current_number_of_snapshots + 1) * 1_000_000
            N = len(self.opponent_manager.agent_pool)
            interval = (N + 1) * 1_000_000
            self.next_snapshot_at += interval
            
            # Print winrates
            log.info("\n--- SelfPlay Opponent Winrates ---")
            log.info("Agent Pool: " + ", ".join(self.opponent_manager.agent_pool))
            log.info("Bot Pool: " + ", ".join(self.opponent_manager.bot_pool))
            log.info("Terminated Pool: " + ", ".join(self.opponent_manager.terminated_pool))
            log.info("Stats:")
            for name, h in self.opponent_manager.stats.items():
                if h['total'] > 0:
                    wr = h['wins'] / h['total']
                    log.info(f"  {name}: {wr*100:.1f}% ({h['wins']}/{h['total']})")
                else:
                    log.info(f"  {name}: No matches yet")
            log.info("----------------------------------")

        # 3. Monitor for best reward and early stopping
        rew = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
        if rew is not None:
            if rew > self.best_reward:
                self.best_reward = rew
                self.model.save(self.best_path)
                if self.verbose:
                    log.info(f"[SelfPlay] New best: {rew:.3f} -> {self.best_path}.zip")

            if rew >= self.target_reward:
                log.info(f"[SelfPlay] Target reached! ep_rew_mean={rew:.3f}")
                return False

        return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--a0-model", default="models/experiment/a0.1_checkpoints/a0.1_snapshot_5000000.zip", help="Path to initial A0.1 model")
    p.add_argument("--steps",    default=30_000_000,  type=int)
    p.add_argument("--resume",   default=None,        help="Resume from a previous checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Ensure model directories exist
    os.makedirs("models/experiment/a1_checkpoints", exist_ok=True)

    # 1. Initialize Opponent Manager
    opponent_manager = OpponentManager(args.a0_model)

    def make_env():
        return SelfPlayEnv(opponent_manager=opponent_manager, phase='A1')

    vec_env = DummyVecEnv([make_env for _ in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    if args.resume:
        log.info(f"[SelfPlay] Resuming from {args.resume}")
        custom_objects = PPO_PARAMS.copy()
        if "policy_kwargs" in custom_objects:
            del custom_objects["policy_kwargs"]
        model = PPO.load(args.resume, env=vec_env, custom_objects=custom_objects)
        model.set_env(vec_env)  # Ensure env is correctly wired
    else:
        log.info(f"[SelfPlay] Loading pretrained A0 weights from {args.a0_model}")
        custom_objects = PPO_PARAMS.copy()
        if "policy_kwargs" in custom_objects:
            del custom_objects["policy_kwargs"]
        model = PPO.load(args.a0_model, env=vec_env, custom_objects=custom_objects)
        model.set_env(vec_env)

    is_resume = bool(args.resume)

    # 2. On resume: reload existing snapshots from disk so opponent pool is intact
    if is_resume:
        ckpt_dir = "models/experiment/a1_checkpoints"
        existing = sorted(
            [f for f in os.listdir(ckpt_dir) if f.startswith("snapshot_") and f.endswith(".zip")],
            key=lambda f: int(f.replace("snapshot_", "").replace(".zip", ""))
        )
        log.info(f"[SelfPlay] Found {len(existing)} existing snapshots on disk — reloading into opponent pool...")
        for fname in existing:
            snap_name = fname.replace(".zip", "")
            snap_path = os.path.join(ckpt_dir, fname)
            opponent_manager.add_opponent(snap_name, snap_path)
            log.info(f"[SelfPlay] Reloaded opponent '{snap_name}'")

    callback = SelfPlayCallback(
        opponent_manager=opponent_manager,
        target_reward=TARGET_REWARD,
        model_dir="models/experiment/",
        envs=vec_env,
        verbose=1
    )

    if is_resume:
        remaining_steps = max(args.steps - model.num_timesteps, 0)
        log.info(f"[SelfPlay] Resuming from step {model.num_timesteps:,} — {remaining_steps:,} steps remaining")
    else:
        remaining_steps = args.steps

    log.info(f"[SelfPlay] Starting — {N_ENVS} envs, {remaining_steps:,} steps to go")
    log.info(f"[SelfPlay] Priority: 50% snapshot category, 50% bot category (Wanderer/Pazzo/Random/Static)")

    model.learn(
        total_timesteps      = remaining_steps,
        callback             = callback,
        reset_num_timesteps  = not is_resume,
        log_interval         = 250_000 // (PPO_PARAMS["n_steps"] * N_ENVS), # log ~ every 250k steps
        progress_bar         = True,
    )

    model.save("models/experiment/a1_final")
    log.info("[SelfPlay] Done -> models/experiment/a1_final.zip")
