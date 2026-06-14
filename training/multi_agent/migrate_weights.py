import os
import sys
import io
import zipfile
import torch

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from training.multi_agent.env_rllib import HaxballRLlibEnv

def main():
    model_dir = "models/single_agent/a1_checkpoints"
    zips = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
    if not zips:
        print("No SB3 checkpoints found in a1_checkpoints.")
        return
        
    zips.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
    
    print("Building RLlib PPO Algorithm to receive weights...")
    ray.init(ignore_reinit_error=True)
    register_env("haxball_multi_v0", lambda config: HaxballRLlibEnv(config))
    
    config = (
        PPOConfig()
        .environment("haxball_multi_v0")
        .framework("torch")
        .training(
            model={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "tanh",
                "vf_share_layers": False,
            }
        )
    )
    algo = config.build()
    
    policy = algo.get_policy("default_policy")
    rllib_model = policy.model
    
    map_keys = {
        "mlp_extractor.policy_net.0.weight": "_hidden_layers.0._model.0.weight",
        "mlp_extractor.policy_net.0.bias": "_hidden_layers.0._model.0.bias",
        "mlp_extractor.policy_net.2.weight": "_hidden_layers.1._model.0.weight",
        "mlp_extractor.policy_net.2.bias": "_hidden_layers.1._model.0.bias",
        "mlp_extractor.value_net.0.weight": "_value_branch_separate.0._model.0.weight",
        "mlp_extractor.value_net.0.bias": "_value_branch_separate.0._model.0.bias",
        "mlp_extractor.value_net.2.weight": "_value_branch_separate.1._model.0.weight",
        "mlp_extractor.value_net.2.bias": "_value_branch_separate.1._model.0.bias",
        "action_net.weight": "_logits._model.0.weight",
        "action_net.bias": "_logits._model.0.bias",
        "value_net.weight": "_value_branch._model.0.weight",
        "value_net.bias": "_value_branch._model.0.bias",
    }
    
    for zip_name in zips:
        zip_path = os.path.join(model_dir, zip_name)
        print(f"\nProcessing {zip_name}...")
        
        sb3_sd = None
        with zipfile.ZipFile(zip_path, "r") as archive:
            for name in archive.namelist():
                if name == "policy.pth":
                    with archive.open(name) as f:
                        sb3_sd = torch.load(io.BytesIO(f.read()), map_location="cpu")
                        break
                        
        if sb3_sd is None:
            print(f"Could not find policy.pth inside {zip_name}. Skipping.")
            continue
            
        new_sd = {}
        for sb3_k, rllib_k in map_keys.items():
            tensor = sb3_sd[sb3_k].clone()
            
            if "0.weight" in sb3_k and tensor.shape[1] == 106:
                col = torch.zeros((tensor.shape[0], 1), dtype=tensor.dtype, device=tensor.device)
                tensor = torch.cat([tensor[:, :4], col, tensor[:, 4:]], dim=1)
                
            new_sd[rllib_k] = tensor
            
        rllib_model.load_state_dict(new_sd)
        
        step_str = zip_name.replace("snapshot_", "").replace(".zip", "")
        out_dir = f"models/multi_agent/rllib_checkpoints/migrated_a1_{step_str}"
        os.makedirs(out_dir, exist_ok=True)
        algo.save(out_dir)
        print(f"Migrated Algorithm saved to: {out_dir}")
        
    print("\nAll weights successfully migrated!")
    
if __name__ == "__main__":
    main()
