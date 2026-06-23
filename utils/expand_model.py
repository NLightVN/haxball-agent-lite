import os
import sys
import torch
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Thêm patch để load được model train trên NumPy 2.x trong môi trường NumPy 1.x
try:
    import numpy.core.numeric
    sys.modules['numpy._core'] = sys.modules['numpy.core']
    sys.modules['numpy._core.numeric'] = sys.modules['numpy.core.numeric']
    sys.modules['numpy._core.multiarray'] = sys.modules.get('numpy.core.multiarray', sys.modules['numpy.core'])
except Exception:
    pass

class DummyEnv(gym.Env):
    def __init__(self, obs_dim=108):
        super().__init__()
        self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([9, 2])
        
    def reset(self, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        
    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-model", default="models/a1_fix_checkpoints/snapshot_2000000.zip")
    parser.add_argument("--new-model", default="models/a2_base.zip")
    args = parser.parse_args()

    print(f"Loading old model: {args.old_model}")
    # Load old model with 106 dims
    old_env = DummyVecEnv([lambda: DummyEnv(obs_dim=106)])
    old_model = PPO.load(args.old_model, env=old_env, custom_objects={'observation_space': old_env.observation_space}, device='cpu')

    print("Creating new model with 108 dims")
    new_env = DummyVecEnv([lambda: DummyEnv(obs_dim=108)])
    
    # Init new model
    policy_kwargs = old_model.policy_kwargs.copy()
    new_model = PPO(
        "MlpPolicy",
        new_env,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        learning_rate=2e-4,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        device='cpu'
    )

    # Copy weights
    old_params = old_model.policy.state_dict()
    new_params = new_model.policy.state_dict()

    # The first layers are usually:
    # mlp_extractor.policy_net.0.weight
    # mlp_extractor.value_net.0.weight

    for key in new_params.keys():
        if key in old_params:
            if old_params[key].shape == new_params[key].shape:
                new_params[key].copy_(old_params[key])
            else:
                print(f"Expanding weights for {key} from {old_params[key].shape} to {new_params[key].shape}")
                # Typically weight shape is (out_features, in_features)
                # old is (256, 106), new is (256, 108)
                old_w = old_params[key]
                new_w = new_params[key]
                
                # Copy old weights
                new_w[:, :old_w.shape[1]] = old_w
                # Initialize new weights (e.g. zeros)
                new_w[:, old_w.shape[1]:] = 0.0

    new_model.policy.load_state_dict(new_params)
    
    # Save the expanded model
    os.makedirs(os.path.dirname(args.new_model), exist_ok=True)
    new_model.save(args.new_model)
    print(f"Successfully saved expanded model to {args.new_model}")

if __name__ == "__main__":
    main()
