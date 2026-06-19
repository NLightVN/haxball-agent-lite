"""
train_a3.py — Phase A3 Multi-Agent Shared Policy Training (3v3).

1 step = 3 ticks.
"""

import argparse
import os
import sys
import glob
import re

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import torch
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from training.envs.env_a3 import A3Env
from training.env import OBS_DIM

N_ENVS          = 4
TOTAL_STEPS     = 80_000_000
CHECKPOINT_FREQ = 100_000
DEFAULT_A01_PATH = "models/a0.1_checkpoints/a0.1_5000000_steps.zip"

class SharedPolicyVecEnv(VecEnv):
    """
    Wraps N HaxballCurriculumEnv (each with 3 agents) into N*3 Gym environments for SB3.
    """
    def __init__(self, env_fns, agents_per_env=3):
        self.envs = [fn() for fn in env_fns]
        self.agents_per_env = agents_per_env
        self.num_envs = len(self.envs) * self.agents_per_env
        
        # Pull observation/action spaces from the first environment
        obs_space = self.envs[0].observation_space
        act_space = self.envs[0].action_space
        super().__init__(self.num_envs, obs_space, act_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs_all, rew_all, done_all, info_all = [], [], [], []
        
        for i, env in enumerate(self.envs):
            # Extract the 3 actions for this environment
            start_idx = i * self.agents_per_env
            env_actions = self.actions[start_idx : start_idx + self.agents_per_env]
            
            # Step the underlying environment with the 3 actions
            obs_list, rew_list, terminated, truncated, info_list = env.step(env_actions)
            done = terminated or truncated
            
            if done:
                # Auto-reset
                obs_list, _ = env.reset()
                
            obs_all.extend(obs_list)
            rew_all.extend(rew_list)
            done_all.extend([done] * self.agents_per_env)
            info_all.extend(info_list)
            
        # Optional: update total_timesteps_elapsed for curriculum
        # We can just increment an internal counter here or rely on callback
        return np.array(obs_all), np.array(rew_all), np.array(done_all), info_all

    def reset(self):
        obs_all = []
        for env in self.envs:
            obs_list, _ = env.reset()
            obs_all.extend(obs_list)
        return np.array(obs_all)

    def close(self):
        for env in self.envs:
            env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(env, attr_name) for env in self.envs]

    def set_attr(self, attr_name, value, indices=None):
        for env in self.envs:
            setattr(env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        results = [getattr(env, method_name)(*method_args, **method_kwargs) for env in self.envs]
        if method_name == "action_masks":
            # Flatten to match self.num_envs (e.g. 12 agents instead of 4 envs)
            flat_results = []
            for res in results:
                for i in range(self.agents_per_env):
                    flat_results.append(res[i])
            return flat_results
        return results
        
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        pass


class A3MonitorCallback(BaseCallback):
    def __init__(self, model_dir, verbose=1):
        super().__init__(verbose)
        self.model_dir = model_dir
        self.best_reward = -np.inf
        self.best_path = os.path.join(model_dir, "a3_best")

    def _on_step(self) -> bool:
        self.training_env.set_attr("total_timesteps_elapsed", self.num_timesteps)

        rew = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
        if rew is None:
            return True

        if rew > self.best_reward:
            self.best_reward = rew
            self.model.save(self.best_path)
            if self.verbose:
                print(f"[A3] ✨ New best reward: {rew:.3f}  → saved to {self.best_path}.zip")

        return True

class DynamicCheckpointCallback(BaseCallback):
    def __init__(self, save_path, name_prefix, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.name_prefix = name_prefix
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
        if self.num_timesteps >= self.next_save_step:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.next_save_step}_steps")
            self.model.save(path)
            if self.verbose:
                print(f"[A3] 💾 Dynamic Checkpoint: Saved model to {path}.zip")
            
            # calculate next
            while self.next_save_step <= self.num_timesteps:
                self.n += 1
                self.next_save_step = self._calc_next_step(self.n)
        return True
class SelfPlayManagerCallback(BaseCallback):
    def __init__(self, checkpoints_dir, total_steps, verbose=1):
        super().__init__(verbose)
        self.checkpoints_dir = checkpoints_dir
        self.total_steps = total_steps
        self.current_opponent_step = -1

    def _on_rollout_start(self) -> None:
        pattern = os.path.join(self.checkpoints_dir, "a3_*_steps.zip")
        files = glob.glob(pattern)
        
        valid_checkpoints = [(0, None)]
        for f in files:
            m = re.search(r'a3_(\d+)_steps\.zip$', f)
            if m:
                valid_checkpoints.append((int(m.group(1)), f))
                
        step_now = self.num_timesteps
        T = max(1_000_000.0, 16_000_000.0 * (1.0 - step_now / self.total_steps) ** 2)
        
        steps, paths = zip(*valid_checkpoints)
        steps = np.array(steps)
        
        diffs = np.abs(steps - step_now)
        probs = np.exp(-diffs / T)
        probs /= np.sum(probs)
        
        sampled_idx = np.random.choice(len(paths), p=probs)
        sampled_step = steps[sampled_idx]
        sampled_path = paths[sampled_idx]
        
        if sampled_step != self.current_opponent_step:
            if self.verbose:
                if sampled_path is None:
                    print(f"[Self-Play] Loading opponent from step 0 (Basic Bots) (T={T:.0f})")
                else:
                    print(f"[Self-Play] Loading opponent from step {sampled_step} (T={T:.0f})")
            try:
                if sampled_path is None:
                    self.training_env.set_attr('override_opponent_policy', None)
                else:
                    opp_model = MaskablePPO.load(sampled_path, custom_objects={"device": "cpu"})
                    self.training_env.set_attr('override_opponent_policy', opp_model)
                self.current_opponent_step = sampled_step
            except Exception as e:
                print(f"[Self-Play] ❌ Error loading opponent: {e}")

    def _on_step(self) -> bool:
        return True


def main():
    parser = argparse.ArgumentParser(description="Train A3 Agent (Shared Policy)")
    parser.add_argument("--continue_step", type=int, default=0, help="Step number to resume from")
    args = parser.parse_args()

    os.makedirs("models/A3_checkpoints", exist_ok=True)
    os.makedirs("tensorboard", exist_ok=True)

    print(f"[A3] Initializing {N_ENVS} parallel 3v3 environments (Total {N_ENVS * 3} agents)...")
    
    def make_env():
        return A3Env()
        
    vec_env = SharedPolicyVecEnv([make_env for _ in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    load_model_path = None
    if args.continue_step > 0:
        load_model_path = os.path.join(ROOT_DIR, f"models/A3_checkpoints/a3_{args.continue_step}_steps.zip")
    else:
        # Load from A0.1 pretrained if starting fresh
        a01_path = os.path.join(ROOT_DIR, DEFAULT_A01_PATH)
        if os.path.exists(a01_path):
            load_model_path = a01_path
        else:
            print(f"[A3] ⚠️  Pretrained not found: {a01_path}")
            print("[A3]    → Training from scratch (no A0.1 weights)")

    print("[A3] Initializing MaskablePPO model...")
    def _make_scratch_model():
        m = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=os.path.join(ROOT_DIR, "tensorboard/A3"),
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.02,
            learning_rate=3e-4,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        )
        with torch.no_grad():
            m.policy.action_net.bias.data[9] += 2.1972
        return m

    if load_model_path and os.path.exists(load_model_path):
        print(f"[A3] Loading model from {load_model_path} to fine-tune...")
        custom_objects = {
            "learning_rate": 5e-5,
            "tensorboard_log": os.path.join(ROOT_DIR, "tensorboard/A3"),
        }
        model = MaskablePPO.load(load_model_path, env=vec_env, custom_objects=custom_objects)
    elif load_model_path:
        # Path specified but file not found
        print(f"[Warning] Could not find file: {load_model_path}")
        print("[A3] Falling back to training from scratch!")
        model = _make_scratch_model()
    else:
        # No pretrained path (a0.1 not found either) → fresh start
        model = _make_scratch_model()

    checkpoint_cb = DynamicCheckpointCallback(
        save_path   = os.path.join(ROOT_DIR, "models/A3_checkpoints/"),
        name_prefix = "a3",
        verbose     = 1,
    )

    monitor_cb = A3MonitorCallback(
        model_dir = os.path.join(ROOT_DIR, "models/"),
        verbose   = 1,
    )

    self_play_cb = SelfPlayManagerCallback(
        checkpoints_dir = os.path.join(ROOT_DIR, "models/A3_checkpoints/"),
        total_steps     = TOTAL_STEPS,
        verbose         = 1,
    )

    remaining_steps = TOTAL_STEPS - args.continue_step if args.continue_step > 0 else TOTAL_STEPS
    print(f"[A3] Starting training — {N_ENVS} envs × 3 agents = {N_ENVS * 3} simultaneous steps.")
    
    model.learn(
        total_timesteps = remaining_steps,
        callback        = CallbackList([checkpoint_cb, monitor_cb, self_play_cb]),
        progress_bar    = True,
        reset_num_timesteps = False,
    )

    final_path = os.path.join(ROOT_DIR, "models/a3_final")
    model.save(final_path)
    print(f"[A3] Done! Final model: {final_path}.zip")

if __name__ == "__main__":
    main()
