import os
import shutil
import torch
from stable_baselines3 import PPO
import sys

# Thêm đường dẫn gốc vào sys.path để import A2Env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.train_a2v1 import A2Env
from training.train_a2v1 import PPO_PARAMS

def expand_model(model_path):
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found.")
        return
        
    backup_path = model_path.replace(".zip", "_backup_108.zip")
    if not os.path.exists(backup_path):
        shutil.copy(model_path, backup_path)
        print(f"Backed up to {backup_path}")

    # Create a dummy env to initialize the new 115-dim architecture
    env = A2Env(opponent_manager=None, teammate_policy=None, phase='A2.0', n_agents=2)
    new_model = PPO("MlpPolicy", env, **PPO_PARAMS)
    
    print(f"Loading old model {model_path} (108 dims)...")
    old_model = PPO.load(backup_path, device='cpu')
    
    old_state = old_model.policy.state_dict()
    new_state = new_model.policy.state_dict()
    
    print("Transferring weights...")
    for name, param in old_state.items():
        if name in new_state:
            if old_state[name].shape == new_state[name].shape:
                new_state[name].copy_(param)
            else:
                print(f"Expanding layer {name}: {old_state[name].shape} -> {new_state[name].shape}")
                old_w = old_state[name]
                new_w = new_state[name]
                # Copy old weights
                new_w[:, :old_w.shape[1]] = old_w
                # Zero out new connections so behavior doesn't change randomly
                new_w[:, old_w.shape[1]:] = 0.0
                
    new_model.policy.load_state_dict(new_state)
    new_model.save(model_path)
    print(f"Done! Overwrote {model_path} with expanded 115-dim model.")

if __name__ == "__main__":
    expand_model("models/a2_base.zip")
    expand_model("models/a2_t1_final.zip")
    expand_model("models/a2_t2_final.zip")
