import argparse
import os
import sys
import glob
import re

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from training.env import HaxballCurriculumEnv
from utils.model_logger import get_model_logger

log = get_model_logger("a2v2_train")

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

# ── Callbacks ──────────────────────────────────────────────────────────────────
class DynamicCheckpointCallback(BaseCallback):
    def __init__(self, team_name, save_path, verbose=1):
        super().__init__(verbose)
        self.team_name = team_name
        self.save_path = save_path
        self.n = 0
        self.next_save_step = self._calc_next_step(self.n)

    def _calc_next_step(self, n):
        val = 500_000 * (1.5 ** n)
        return round(val / 100_000) * 100_000

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        # catch up to current step
        while self.next_save_step <= self.num_timesteps:
            self.n += 1
            self.next_save_step = self._calc_next_step(self.n)

    def _on_step(self) -> bool:
        # Pass timesteps to the environment
        self.training_env.set_attr('total_timesteps_elapsed', self.num_timesteps)

        if self.num_timesteps >= self.next_save_step:
            path = os.path.join(self.save_path, f"snapshot_{self.next_save_step}")
            self.model.save(path)
            if self.verbose:
                log.info(f"[{self.team_name}] 💾 Dynamic Checkpoint: Saved model to {path}.zip")
            
            # calculate next
            while self.next_save_step <= self.num_timesteps:
                self.n += 1
                self.next_save_step = self._calc_next_step(self.n)
        return True


class SelfPlayManagerCallback(BaseCallback):
    def __init__(self, team_name, checkpoints_dir, total_steps, verbose=1):
        super().__init__(verbose)
        self.team_name = team_name
        self.checkpoints_dir = checkpoints_dir
        self.total_steps = total_steps
        self.current_opponent_step = -1

    def _on_rollout_start(self) -> None:
        pattern = os.path.join(self.checkpoints_dir, "snapshot_*.zip")
        files = glob.glob(pattern)
        
        bot_types = ['Pazzo', 'Wanderer', 'Defender', 'Attacker']
        chosen_bots = np.random.choice(bot_types, 2, replace=False)
        
        valid_checkpoints = [(0, chosen_bots[0]), (0, chosen_bots[1])]
        for f in files:
            m = re.search(r'snapshot_(\d+)\.zip$', f)
            if m:
                valid_checkpoints.append((int(m.group(1)), f))
                
        step_now = self.num_timesteps
        T = max(1_000_000.0, 16_000_000.0 * (1.0 - step_now / self.total_steps) ** 2)
        
        if len(valid_checkpoints) == 1:
            sampled_idx = 0
        else:
            steps, paths = zip(*valid_checkpoints)
            steps = np.array(steps)
            
            diffs = np.abs(steps - step_now)
            probs = np.exp(-diffs / T)
            probs /= np.sum(probs)
            
            sampled_idx = np.random.choice(len(paths), p=probs)
            
        sampled_step = valid_checkpoints[sampled_idx][0]
        sampled_path = valid_checkpoints[sampled_idx][1]
        
        if sampled_step != self.current_opponent_step or sampled_path in bot_types:
            try:
                if sampled_path in bot_types:
                    self.training_env.set_attr('forced_opponent_type', sampled_path)
                    self.training_env.set_attr('opponent_policy', None)
                    if self.verbose:
                        log.info(f"[{self.team_name}] [Self-Play] Loaded Bot '{sampled_path}' as opponent (T={T:.0f})")
                else:
                    opp_model = PPO.load(sampled_path, custom_objects={"device": "cpu"})
                    self.training_env.set_attr('forced_opponent_type', 'Trained')
                    self.training_env.set_attr('opponent_policy', opp_model)
                    if self.verbose:
                        log.info(f"[{self.team_name}] [Self-Play] Loaded opponent from step {sampled_step} (T={T:.0f})")
                self.current_opponent_step = sampled_step
            except Exception as e:
                log.info(f"[{self.team_name}] [Self-Play] ❌ Error loading opponent: {e}")

    def _on_step(self) -> bool:
        return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="models/a2_t1_checkpoints/snapshot_1000000.zip", help="Path to initial expanded model")
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

    # Đảm bảo snapshot_1000000.zip có trong pool của T2 (T1 đã có do path mặc định)
    import shutil
    if os.path.exists(args.base_model):
        t2_target = "models/a2_t2_checkpoints/snapshot_1000000.zip"
        if args.base_model != t2_target and not os.path.exists(t2_target):
            shutil.copy(args.base_model, t2_target)
            log.info(f"Copied {args.base_model} to {t2_target}")
    else:
        log.warning(f"Base model {args.base_model} not found. Training from scratch.")

    def make_env_t1():
        # Env for training T1 (n_agents=2). 
        # Opponent policy will be set by SelfPlayManagerCallback
        return HaxballCurriculumEnv(phase='A2.0', n_agents=2)
        
    def make_env_t2():
        # Env for training T2 (n_agents=1).
        # Opponent policy will be set by SelfPlayManagerCallback
        return HaxballCurriculumEnv(phase='A2.0', n_agents=1)

    vec_env_t1 = DummyVecEnv([make_env_t1 for _ in range(N_ENVS)])
    vec_env_t1 = VecMonitor(vec_env_t1)
    
    vec_env_t2 = DummyVecEnv([make_env_t2 for _ in range(N_ENVS)])
    vec_env_t2 = VecMonitor(vec_env_t2)

    log.info(f"Loading initial weights from {args.base_model}")
    if os.path.exists(args.base_model):
        model_t1 = PPO.load(args.base_model, env=vec_env_t1, custom_objects=PPO_PARAMS)
        model_t2 = PPO.load(args.base_model, env=vec_env_t2, custom_objects=PPO_PARAMS)
    else:
        model_t1 = PPO("MlpPolicy", vec_env_t1, **PPO_PARAMS)
        model_t2 = PPO("MlpPolicy", vec_env_t2, **PPO_PARAMS)
        
    model_t1.set_env(vec_env_t1)
    model_t2.set_env(vec_env_t2)

    # Set initial teammate policy for T1 envs dynamically so it uses the current model_t1 weights
    for env in vec_env_t1.envs:
        env.teammate_policy = model_t1

    # Callbacks for T1 (Trains against T2's checkpoints)
    t1_ckpt_cb = DynamicCheckpointCallback("T1", "models/a2_t1_checkpoints", verbose=1)
    t1_opp_cb  = SelfPlayManagerCallback("T1", "models/a2_t2_checkpoints", args.steps, verbose=1)

    # Callbacks for T2 (Trains against T1's checkpoints)
    t2_ckpt_cb = DynamicCheckpointCallback("T2", "models/a2_t2_checkpoints", verbose=1)
    t2_opp_cb  = SelfPlayManagerCallback("T2", "models/a2_t1_checkpoints", args.steps, verbose=1)

    total_steps = args.steps
    chunk = args.chunk_steps
    
    log.info(f"Starting Alternating Training: {total_steps:,} steps total. Chunk size: {chunk:,}")
    
    while model_t1.num_timesteps < total_steps or model_t2.num_timesteps < total_steps:
        # Train T1 (Team 1, 2 players)
        log.info(f"\n--- Training T1 (Step {model_t1.num_timesteps:,} / {total_steps:,}) ---")
        for env in vec_env_t1.envs:
            env.teammate_policy = model_t1
        
        model_t1.learn(
            total_timesteps=chunk,
            callback=[t1_ckpt_cb, t1_opp_cb],
            reset_num_timesteps=False,
            log_interval=10,
            progress_bar=True,
        )
        model_t1.save("models/a2_t1_final")
        
        # Train T2 (Team 2, 1 player)
        log.info(f"\n--- Training T2 (Step {model_t2.num_timesteps:,} / {total_steps:,}) ---")
        model_t2.learn(
            total_timesteps=chunk,
            callback=[t2_ckpt_cb, t2_opp_cb],
            reset_num_timesteps=False,
            log_interval=10,
            progress_bar=True,
        )
        model_t2.save("models/a2_t2_final")

    log.info("Finished Alternating Training!")
