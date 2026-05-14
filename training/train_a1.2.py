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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.env import HaxballCurriculumEnv

# ── Config ─────────────────────────────────────────────────────────────────────
N_ENVS          = 8
FIRST_SNAPSHOT_AT = 3_000_000
CHECKPOINT_FREQ = 2_000_000
TARGET_REWARD   = 15.0
BOT_TYPES       = ('Defender', 'Attacker', 'Hybrid', 'Random')

PPO_PARAMS = dict(
    n_steps      = 512,
    batch_size   = 256,
    n_epochs     = 8,
    gamma        = 0.99,
    gae_lambda   = 0.98,
    ent_coef     = 0.02,
    learning_rate= 2e-4,     # slightly lower for fine-tuning stability
    clip_range   = 0.2,
    vf_coef      = 0.5,
    max_grad_norm= 0.5,
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

        if os.path.exists(initial_model_path):
            self.add_opponent("A0_baseline", initial_model_path)

        
    def add_opponent(self, name, model_path):
        print(f"[SelfPlay] Loading opponent '{name}' from {model_path} into memory...")
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
                if snapshot_winrate < 0.075:
                    print(f"\n[SelfPlay] Opponent '{name}' winrate {snapshot_winrate:.3f} < 0.075 after {h['total']} matches. Moving to terminated pool.\n")
                    if name in self.agent_pool:
                        self.agent_pool.remove(name)
                    self.terminated_pool.append(name)
                    # Reset indices to avoid out of bounds
                    self.agent_idx = 0
                    self.terminated_idx = 0
        
    def sample_opponent(self):
        has_agents = len(self.agent_pool) > 0
        has_bots = len(self.bot_pool) > 0

        if not has_agents and not has_bots:
            return None, None

        # 50% agent snapshot category, 50% bot category.
        # If one category is missing, fall back to the other.
        choose_bot = False
        if has_agents and has_bots:
            choose_bot = np.random.rand() >= 0.5
        elif has_bots:
            choose_bot = True

        if choose_bot:
            bot_name = self.bot_pool[int(np.random.randint(0, len(self.bot_pool)))]
            return None, bot_name

        chosen_name = self.agent_pool[self.agent_idx % len(self.agent_pool)]
        self.agent_idx = (self.agent_idx + 1) % len(self.agent_pool)
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
        obs, reward, terminated, truncated, info = super().step(action)
        
        if terminated or truncated:
            # Determine match result
            # team_id is 1 (RED) or 2 (BLUE). scores array is [RED, BLUE]
            agent_score = self.scores[self.team_id - 1]
            opp_score = self.scores[2 - self.team_id]
            
            if self.episode_type == 'opponent' and self.current_opponent_name is not None:
                if agent_score != opp_score:
                    agent_won = (agent_score > opp_score)
                    self.opponent_manager.update_result(self.current_opponent_name, agent_won)
                    
        return obs, reward, terminated, truncated, info

# ── Combined Callback ──────────────────────────────────────────────────────────
class SelfPlayCallback(BaseCallback):
    def __init__(self, opponent_manager, target_reward, model_dir, envs, save_freq, verbose=1):
        super().__init__(verbose)
        self.opponent_manager = opponent_manager
        self.target_reward = target_reward
        self.model_dir = model_dir
        self.envs = envs
        self.checkpoint_interval = save_freq
        self.next_snapshot_at = FIRST_SNAPSHOT_AT
        self.best_reward = -np.inf
        self.best_path = os.path.join(model_dir, "a1.2_best")
        
    def _on_training_start(self) -> None:
        # Do not store snapshot_0; training starts directly from the bootstrap model.
        pass
        
    def _on_step(self) -> bool:
        # 1. Sync timesteps so env curriculum sees current progress
        for env in self.envs.envs:
            env.total_timesteps_elapsed = self.num_timesteps
            
        # 2. Checkpoint and register new opponent
        while self.next_snapshot_at <= self.num_timesteps:
            snapshot_name = f"snapshot_{self.num_timesteps}"
            path = os.path.join(self.model_dir, "a1.2_checkpoints", f"{snapshot_name}.zip")
            self.model.save(path)
            
            if self.verbose:
                print(f"[SelfPlay] Saved snapshot to {path}")
            
            self.opponent_manager.add_opponent(snapshot_name, path)
            self.next_snapshot_at += self.checkpoint_interval
            
            # Print winrates
            print("\n--- SelfPlay Opponent Winrates ---")
            print("Agent Pool: " + ", ".join(self.opponent_manager.agent_pool))
            print("Bot Pool: " + ", ".join(self.opponent_manager.bot_pool))
            print("Terminated Pool: " + ", ".join(self.opponent_manager.terminated_pool))
            print("Stats:")
            for name, h in self.opponent_manager.stats.items():
                if h['total'] > 0:
                    wr = h['wins'] / h['total']
                    print(f"  {name}: {wr*100:.1f}% ({h['wins']}/{h['total']})")
                else:
                    print(f"  {name}: No matches yet")
            print("----------------------------------\n")

        # 3. Monitor for best reward and early stopping
        rew = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
        if rew is not None:
            if rew > self.best_reward:
                self.best_reward = rew
                self.model.save(self.best_path)
                if self.verbose:
                    print(f"[SelfPlay] New best: {rew:.3f} -> {self.best_path}.zip")

            if rew >= self.target_reward:
                print(f"\n[SelfPlay] Target reached! ep_rew_mean={rew:.3f}")
                return False

        return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--a0-model", default="models/a0.1_checkpoints/a0.1_snapshot_500000.zip", help="Path to initial A0.1 model")
    p.add_argument("--steps",    default=16_000_000,  type=int)
    p.add_argument("--resume",   default=None,        help="Resume from a previous checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Ensure model directories exist
    os.makedirs("models/a1.2_checkpoints", exist_ok=True)

    # 1. Initialize Opponent Manager
    opponent_manager = OpponentManager(args.a0_model)

    def make_env():
        return SelfPlayEnv(opponent_manager=opponent_manager, phase='A1.2')

    vec_env = DummyVecEnv([make_env for _ in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    if args.resume:
        print(f"[SelfPlay] Resuming from {args.resume}")
        model = PPO.load(args.resume, env=vec_env)
        model.set_env(vec_env)  # Ensure env is correctly wired
    else:
        print(f"[SelfPlay] Loading pretrained A0 weights from {args.a0_model}")
        model = PPO.load(args.a0_model, env=vec_env, custom_objects=PPO_PARAMS)
        model.set_env(vec_env)

    callback = SelfPlayCallback(
        opponent_manager=opponent_manager,
        target_reward=TARGET_REWARD,
        model_dir="models/",
        envs=vec_env,
        save_freq=CHECKPOINT_FREQ,
        verbose=1
    )

    is_resume = bool(args.resume)
    if is_resume:
        remaining_steps = max(args.steps - model.num_timesteps, 0)
        print(f"[SelfPlay] Resuming from step {model.num_timesteps:,} — {remaining_steps:,} steps remaining")
    else:
        remaining_steps = args.steps

    print(f"[SelfPlay] Starting — {N_ENVS} envs, {remaining_steps:,} steps to go")
    print(f"[SelfPlay] Priority: 50% snapshot category, 50% bot category (Defender/Attacker/Hybrid/Random)")

    model.learn(
        total_timesteps      = remaining_steps,
        callback             = callback,
        reset_num_timesteps  = not is_resume,
        log_interval         = 250_000 // (PPO_PARAMS["n_steps"] * N_ENVS), # log ~ every 250k steps
        progress_bar         = True,
    )

    model.save("models/a1.2_final")
    print("\n[SelfPlay] Done -> models/a1.2_final.zip")
