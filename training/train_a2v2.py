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
        valid_checkpoints = [(0, "BOTS")]
        for f in files:
            m = re.search(r'snapshot_(\d+)\.zip$', f)
            if m:
                valid_checkpoints.append((int(m.group(1)), f))
                
        step_now = self.num_timesteps
        T = max(1_000_000.0, 16_000_000.0 * (1.0 - step_now / self.total_steps) ** 2)
        
        if len(valid_checkpoints) == 1:
            probs = [1.0]
        else:
            steps, paths = zip(*valid_checkpoints)
            steps = np.array(steps)
            
            diffs = np.abs(steps - step_now)
            probs = np.exp(-diffs / T)
            probs /= np.sum(probs)
            
        loaded_models = {}
        log_messages = []
        
        for i in range(self.training_env.num_envs):
            sampled_idx = 0 if len(valid_checkpoints) == 1 else np.random.choice(len(valid_checkpoints), p=probs)
            sampled_step = valid_checkpoints[sampled_idx][0]
            sampled_path = valid_checkpoints[sampled_idx][1]
            
            try:
                if sampled_path == "BOTS":
                    chosen_bots = np.random.choice(bot_types, 2, replace=False).tolist()
                    self.training_env.set_attr('forced_opponent_type', chosen_bots, indices=[i])
                    self.training_env.set_attr('opponent_policy', None, indices=[i])
                    log_messages.append(f"Env {i}: Bots {chosen_bots}")
                else:
                    if sampled_path not in loaded_models:
                        loaded_models[sampled_path] = PPO.load(sampled_path, custom_objects={"device": "cpu"})
                    
                    self.training_env.set_attr('forced_opponent_type', ['Trained', 'Trained'], indices=[i])
                    self.training_env.set_attr('opponent_policy', loaded_models[sampled_path], indices=[i])
                    log_messages.append(f"Env {i}: Step {sampled_step}")
            except Exception as e:
                log.info(f"[{self.team_name}] [Self-Play] ❌ Error loading opponent for Env {i}: {e}")
                
        if self.verbose:
            log.info(f"[{self.team_name}] [Self-Play] Rolled opponents (T={T:.0f}) -> {', '.join(log_messages)}")

    def _on_step(self) -> bool:
        return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="models/a2_t1_checkpoints/snapshot_1000000.zip", help="Path to initial expanded model")
    p.add_argument("--steps",    default=30_000_000,  type=int)
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

    os.makedirs("models/a2v2_checkpoints", exist_ok=True) # Symmetrical Self-Play directory

    # Đảm bảo snapshot_1000000.zip có trong pool của a2v2_checkpoints
    import shutil
    if os.path.exists(args.base_model):
        a2v2_target = "models/a2v2_checkpoints/snapshot_1.zip"
        if args.base_model != a2v2_target and not os.path.exists(a2v2_target):
            shutil.copy(args.base_model, a2v2_target)
            log.info(f"Copied {args.base_model} to {a2v2_target}")
    else:
        log.warning(f"Base model {args.base_model} not found. Training from scratch.")

    def make_env():
        # Env for Symmetrical Self-Play (A2v2, 2vs2).
        return HaxballCurriculumEnv(phase='A2v2', n_agents=2)

    vec_env = DummyVecEnv([make_env for _ in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)
    
    log.info(f"Loading initial weights from {args.base_model}")
    if os.path.exists(args.base_model):
        model = PPO.load(args.base_model, env=vec_env, custom_objects=PPO_PARAMS)
        model.num_timesteps = 0  # Force restart from step 0 for A2v2
        log.info("Model loaded. Starting A2v2 training from step 0.")
    else:
        model = PPO("MlpPolicy", vec_env, **PPO_PARAMS)
        
    model.set_env(vec_env)

    # Set initial teammate policy dynamically so it uses the current model weights
    for env in vec_env.envs:
        env.teammate_policy = model

    ckpt_cb = DynamicCheckpointCallback("A2v2", "models/a2v2_checkpoints", verbose=1)
    opp_cb  = SelfPlayManagerCallback("A2v2", "models/a2v2_checkpoints", args.steps, verbose=1)

    total_steps = args.steps
    
    log.info(f"Starting Symmetrical Self-Play Training: {total_steps:,} steps total.")
    
    model.learn(
        total_timesteps=total_steps,
        callback=[ckpt_cb, opp_cb],
        reset_num_timesteps=False,
        log_interval=10,
        progress_bar=True,
    )
    
    model.save("models/a2v2_final")
    log.info("Finished Symmetrical Self-Play Training!")
