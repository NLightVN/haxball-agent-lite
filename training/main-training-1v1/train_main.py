"""
train_main.py — MAIN_1V1 Training.

Episode types:
  • PRECISION : small goal (0.3–0.6×), no opponent → aim accuracy
  • OPPONENT  : goal 1.4×→0.6× over 2M steps, opponent = Sampled from self-play pool

Usage:
    python training/main-training-1v1/train_main.py
"""

import argparse
import collections
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
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

log = get_model_logger("main_1v1_train")

# ── Config ─────────────────────────────────────────────────────────────────────
N_ENVS          = 8
TARGET_REWARD   = 15.0
BOT_TYPES       = ('Pazzo', 'Random', 'Static', 'Wanderer')

PPO_PARAMS = dict(
    n_steps      = 2048,
    batch_size   = 512,
    n_epochs     = 10,
    gamma        = 0.99,
    gae_lambda   = 0.95,
    ent_coef     = 0.02,
    learning_rate= 3e-4,     # slightly lower for fine-tuning stability
    clip_range   = 0.2,
    vf_coef      = 0.5,
    max_grad_norm= 0.5,
    verbose      = 1,
    policy_kwargs= dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
)

# ── Self-Play Opponent Manager ────────────────────────────────────────────────
class OpponentManager:
    def __init__(self, initial_model_path):
        self.opponents = {}  # name -> policy
        self.agent_pool = []
        self.bot_pool = list(BOT_TYPES)
        self.terminated_pool = []
        self.exploiter_pool = []
        self.exploiter_dir = "models/main-training-1v1/exploiter_checkpoints"
        self.last_exploiter_refresh = 0
        
        # Stats per opponent: deque of booleans (True if opponent won)
        self.stats = {} # name -> deque(maxlen=200)

    def add_opponent(self, name, model_path):
        log.info(f"[SelfPlay] Loading opponent '{name}' from {model_path} into memory...")
        policy = PPO.load(model_path, device='cpu')
        self.opponents[name] = policy
        self.stats[name] = collections.deque(maxlen=200)
        self.agent_pool.append(name)
            
    def update_result(self, name, agent_won):
        if name in self.stats:
            self.stats[name].append(not agent_won)
                
            history = self.stats[name]
            if name in self.agent_pool and len(history) >= 20:
                snapshot_winrate = sum(history) / len(history)
                if snapshot_winrate < 0.05:
                    log.info(f"[SelfPlay] Opponent '{name}' winrate {snapshot_winrate:.3f} < 0.05 over last {len(history)} matches. Moving to terminated pool.")
                    if name in self.agent_pool:
                        self.agent_pool.remove(name)
                    self.terminated_pool.append(name)

    def _refresh_exploiters(self):
        import time
        if not os.path.exists(self.exploiter_dir):
            return
        files = sorted([f for f in os.listdir(self.exploiter_dir) if f.endswith(".zip")])
        for f in files:
            name = f"exploiter_{f.replace('.zip', '')}"
            if name not in self.opponents:
                path = os.path.join(self.exploiter_dir, f)
                log.info(f"[SelfPlay] Found new Exploiter '{name}' -> Loading...")
                try:
                    policy = PPO.load(path, device='cpu')
                    self.opponents[name] = policy
                    self.exploiter_pool.append(name)
                except Exception as e:
                    log.error(f"[SelfPlay] Failed to load exploiter {name}: {e}")

    def sample_opponent(self, current_step=0):
        import time
        now = time.time()
        if now - self.last_exploiter_refresh > 60:
            self._refresh_exploiters()
            self.last_exploiter_refresh = now
            
        r = np.random.rand()
        has_exploiter = len(self.exploiter_pool) > 0
        has_pfsp = len(self.agent_pool) > 0
        
        # 10% Bot, 20% Exploiter, 70% PFSP
        bot_prob = 0.10
        exploiter_prob = 0.20 if has_exploiter else 0.0
        pfsp_prob = 0.70 if has_pfsp else 0.0
        
        total = bot_prob + exploiter_prob + pfsp_prob
        bot_prob /= total
        exploiter_prob /= total
        pfsp_prob /= total
        
        if r < bot_prob:
            valid_bots = ['Pazzo', 'Random', 'Static']
            if current_step >= 10_000_000:
                valid_bots.append('Wanderer')
            bot_name = np.random.choice(valid_bots)
            return None, bot_name
        elif r < bot_prob + exploiter_prob:
            chosen_name = np.random.choice(self.exploiter_pool)
            return self.opponents[chosen_name], chosen_name
        else:
            weights = []
            for name in self.agent_pool:
                h = self.stats.get(name, collections.deque(maxlen=200))
                if len(h) == 0:
                    w = 1.0
                else:
                    opp_winrate = sum(h) / len(h)
                    w = max(0.05, opp_winrate) # PFSP weight = loss rate
                weights.append(w)
            
            weights = np.array(weights)
            weights /= weights.sum()
            chosen_name = np.random.choice(self.agent_pool, p=weights)
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
            policy, name = self.opponent_manager.sample_opponent(self.total_timesteps_elapsed)
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
        self.best_path = os.path.join(model_dir, "main_1v1_best")
        
    def _on_training_start(self) -> None:
        N = len(self.opponent_manager.agent_pool)
        self.next_snapshot_at = (N + 1) * 1_000_000
        if N > 0:
            log.info(f"[SelfPlay] Resume: {N} existing snapshots, next snapshot at step {self.next_snapshot_at:,}")
        
    def _on_step(self) -> bool:
        # 1. Sync timesteps so env curriculum sees current progress
        for env in self.envs.envs:
            env.total_timesteps_elapsed = self.num_timesteps
            
        # 2. Checkpoint and register new opponent
        while self.next_snapshot_at <= self.num_timesteps:
            snapshot_name = f"snapshot_{self.num_timesteps}"
            path = os.path.join(self.model_dir, "main_1v1_checkpoints", f"{snapshot_name}.zip")
            self.model.save(path)
            
            if self.verbose:
                log.info(f"[SelfPlay] Saved snapshot to {path}")
            
            self.opponent_manager.add_opponent(snapshot_name, path)
            
            self.next_snapshot_at += 1_000_000
            
            # Print winrates
            log.info("\n--- SelfPlay Opponent Winrates ---")
            log.info("Agent Pool: " + ", ".join(self.opponent_manager.agent_pool))
            log.info("Bot Pool: " + ", ".join(self.opponent_manager.bot_pool))
            log.info("Terminated Pool: " + ", ".join(self.opponent_manager.terminated_pool))
            log.info("Stats:")
            for name, h in self.opponent_manager.stats.items():
                if len(h) > 0:
                    wins = sum(h)
                    total = len(h)
                    wr = wins / total
                    log.info(f"  {name}: {wr*100:.1f}% ({wins}/{total})")
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
    p.add_argument("--a0-model", default=None, help="Path to initial A0.1 model (leave empty to train from scratch)")
    p.add_argument("--steps",    default=30_000_000,  type=int)
    p.add_argument("--resume",   default=None,        help="Resume from a previous checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Ensure model directories exist
    os.makedirs("models/main-training-1v1/main_1v1_checkpoints", exist_ok=True)

    # 1. Initialize Opponent Manager
    opponent_manager = OpponentManager(args.a0_model)

    def make_env():
        return SelfPlayEnv(opponent_manager=opponent_manager, phase='MAIN_1V1')

    vec_env = DummyVecEnv([make_env for _ in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    if args.resume:
        log.info(f"[SelfPlay] Resuming from {args.resume}")
        custom_objects = PPO_PARAMS.copy()
        if "policy_kwargs" in custom_objects:
            del custom_objects["policy_kwargs"]
        custom_objects["tensorboard_log"] = "./tensorboard/main_1v1"
        model = PPO.load(args.resume, env=vec_env, custom_objects=custom_objects)
        model.set_env(vec_env)  # Ensure env is correctly wired
    elif args.a0_model and os.path.exists(args.a0_model):
        log.info(f"[SelfPlay] Loading pretrained A0 weights from {args.a0_model}")
        custom_objects = PPO_PARAMS.copy()
        if "policy_kwargs" in custom_objects:
            del custom_objects["policy_kwargs"]
        custom_objects["tensorboard_log"] = "./tensorboard/main_1v1"
        model = PPO.load(args.a0_model, env=vec_env, custom_objects=custom_objects)
        model.set_env(vec_env)
    else:
        log.info("[SelfPlay] Starting training from scratch (no pretrained weights)")
        params = PPO_PARAMS.copy()
        params["tensorboard_log"] = "./tensorboard/main_1v1"
        model = PPO("MlpPolicy", env=vec_env, **params)

    is_resume = bool(args.resume)

    # 2. On resume: reload existing snapshots from disk so opponent pool is intact
    if is_resume:
        ckpt_dir = "models/main-training-1v1/main_1v1_checkpoints"
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
        model_dir="models/main-training-1v1/",
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

    model.save("models/main-training-1v1/main_1v1_final")
    log.info("[SelfPlay] Done -> models/main-training-1v1/main_1v1_final.zip")
