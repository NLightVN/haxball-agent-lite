import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from training.multi_agent.env_rllib import HaxballRLlibEnv
from training.multi_agent.rllib_callbacks import HaxballCallbacks

def env_creator(env_config):
    return HaxballRLlibEnv(env_config)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    # In full Self-Play, all agents use the same policy
    return "learning_agent"

def main():
    # Initialize Ray
    ray.init()
    
    # Register the environment
    register_env("haxball_multi_v0", env_creator)
    
    # Configure PPO
    config = (
        PPOConfig()
        .environment("haxball_multi_v0")
        .framework("torch")
        .rollouts(
            num_rollout_workers=2, # Keep low for testing/local
            batch_mode="complete_episodes", # VERY IMPORTANT: Ensures we get full matches
        )
        .training(
            train_batch_size=8000,
            sgd_minibatch_size=1000,
            num_sgd_iter=10,
            lr=3e-4,
            model={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "tanh",
                "vf_share_layers": False,
            }
        )
        .multi_agent(
            policies={"learning_agent"},
            policy_mapping_fn=policy_mapping_fn,
        )
        .callbacks(HaxballCallbacks)
    )
    
    print("Building RLlib PPO Algorithm...")
    algo = config.build()
    
    os.makedirs("models/multi_agent/rllib_checkpoints", exist_ok=True)
    
    print("Starting training loop...")
    for i in range(100000):
        result = algo.train()
        
        # RLlib 2.x stores custom metrics in 'custom_metrics', but standard rewards are in:
        # result["env_runners"]["episode_reward_mean"] or result["episode_reward_mean"] depending on version
        rew_mean = result.get("episode_reward_mean", result.get("env_runners", {}).get("episode_reward_mean", 0))
        len_mean = result.get("episode_len_mean", result.get("env_runners", {}).get("episode_len_mean", 0))
        
        print(f"Iteration {i}: reward_mean={rew_mean:.2f}, len_mean={len_mean:.2f}")
              
        if i % 10 == 0:
            checkpoint_dir = algo.save("models/multi_agent/rllib_checkpoints")
            print(f"Saved checkpoint to {checkpoint_dir}")

if __name__ == "__main__":
    main()
