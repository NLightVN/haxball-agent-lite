import os
import sys
import time

# Prevent TensorFlow DLL issues on Windows - must be set before any ray import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from training.multi_agent.env_rllib import HaxballRLlibEnv
from training.multi_agent.rllib_callbacks import HaxballCallbacks

from ray.rllib.policy.policy import PolicySpec, Policy
from training.multi_agent.bot_policy import BotPolicy
import random

def env_creator(env_config):
    return HaxballRLlibEnv(env_config)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id in ["red_0", "red_1"]:
        return "learning_agent"
        
    ts = worker.global_vars.get("timesteps_total", 0)
    match_mode = "2v2"
    if hasattr(episode, "user_data") and "match_mode" in episode.user_data:
        match_mode = episode.user_data["match_mode"]

    if "opp_policy" in episode.user_data:
        return episode.user_data["opp_policy"]

    if ts < 10_000_000:
        if match_mode == "1v1":
            chosen = random.choice(["a1_snap_1000000", "a1_snap_3000000", "a1_snap_6000000", "a1_snap_10000000", "a1_snap_15000000"])
        else:
            chosen = "bot_policy"
    else:
        if match_mode == "1v1":
            valid_a2 = [f"a2_snap_{s}" for s in ["1M", "3M", "5M", "7M", "10M", "13M", "17M", "21M", "25M", "30M"] if int(s[:-1])*1000000 <= ts]
            valid_a1 = ["a1_snap_1000000", "a1_snap_3000000", "a1_snap_6000000", "a1_snap_10000000", "a1_snap_15000000"]
            chosen = random.choice(valid_a1 + valid_a2)
        else:
            valid_a2 = [f"a2_snap_{s}" for s in ["1M", "3M", "5M", "7M", "10M", "13M", "17M", "21M", "25M", "30M"] if int(s[:-1])*1000000 <= ts]
            if not valid_a2:
                chosen = "learning_agent"
            elif random.random() < (3.0 / 8.0):
                chosen = random.choice(valid_a2)
            else:
                pfsp_rates = worker.global_vars.get("pfsp_winrates", {})
                weights = [pfsp_rates.get(p, 1.0) for p in valid_a2]
                sum_w = sum(weights)
                if sum_w == 0:
                    chosen = random.choice(valid_a2)
                else:
                    probs = [w / sum_w for w in weights]
                    chosen = random.choices(valid_a2, weights=probs, k=1)[0]
    
    episode.user_data["opp_policy"] = chosen
    return chosen

def main():
    # Initialize Ray
    ray.init(
        runtime_env={"env_vars": {
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "TF_ENABLE_ONEDNN_OPTS": "0",
        }},
        logging_level="ERROR",
    )
    
    # Register the environment
    register_env("haxball_multi_v0", env_creator)
    
    policies_dict = {
        "learning_agent": PolicySpec(),
        "bot_policy": PolicySpec(policy_class=BotPolicy),
    }
    for s in ["1000000", "3000000", "6000000", "10000000", "15000000"]:
        policies_dict[f"a1_snap_{s}"] = PolicySpec()
    for s in ["1M", "3M", "5M", "7M", "10M", "13M", "17M", "21M", "25M", "30M"]:
        policies_dict[f"a2_snap_{s}"] = PolicySpec()

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
            policies=policies_dict,
            policies_to_train=["learning_agent"],
            policy_mapping_fn=policy_mapping_fn,
        )
        .callbacks(HaxballCallbacks)
        .update_from_dict({
            "extra_python_environs_for_worker": {
                "RLLIB_TEST_NO_TF_IMPORT": "1"
            }
        })
    )
    
    print("Building RLlib PPO Algorithm for A2...")
    algo = config.build()
    
    print("Loading A1 snapshot weights...")
    for s in ["1000000", "3000000", "6000000", "10000000", "15000000"]:
        ckpt_path = f"models/multi_agent/rllib_checkpoints/migrated_a1_{s}/policies/default_policy"
        if os.path.exists(ckpt_path):
            pol = Policy.from_checkpoint(ckpt_path)
            algo.get_policy(f"a1_snap_{s}").set_state(pol.get_state())
            print(f"  Loaded A1 {s}")

    print("Loading A1 finetune snapshot into learning_agent...")
    finetune_ckpt = "models/multi_agent/rllib_checkpoints/a1_finetune_snapshot_500000/policies/learning_agent"
    if os.path.exists(finetune_ckpt):
        pol = Policy.from_checkpoint(finetune_ckpt)
        algo.get_policy("learning_agent").set_state(pol.get_state())
        print("  Loaded a1_finetune_snapshot_500000 into learning_agent")

    os.makedirs("models/multi_agent/rllib_checkpoints/a2", exist_ok=True)
    
    print("Starting A2 training loop...")
    print(f"{'Iter':>6} | {'Total Steps':>12} | {'Rew/ep':>8} | {'Len/ep':>8} | {'#Episodes':>10} | {'Time/iter':>10}")
    print("-" * 75)
    t_last = time.time()
    
    a2_milestones = [1, 3, 5, 7, 10, 13, 17, 21, 25, 30]
    a2_milestone_steps = [m * 1000000 for m in a2_milestones]
    next_milestone_idx = 0

    for i in range(100000):
        result = algo.train()
        t_now = time.time()
        dt = t_now - t_last
        t_last = t_now
        
        ts = result["timesteps_total"]
        env_r = result.get("env_runners", result)
        rew_mean = env_r.get("episode_reward_mean", result.get("episode_reward_mean", float('nan')))
        len_mean = env_r.get("episode_len_mean", result.get("episode_len_mean", float('nan')))
        n_ep = env_r.get("episodes_this_iter", result.get("episodes_this_iter", 0))
        
        rew_str = f"{rew_mean:8.2f}" if rew_mean == rew_mean else "       -"
        len_str = f"{len_mean:8.1f}" if len_mean == len_mean else "       -"
        print(f"{i:6d} | {ts:>12,} | {rew_str} | {len_str} | {n_ep:>10} | {dt:8.1f}s")
              
        if i > 0 and i % 10 == 0:
            checkpoint_dir = algo.save("models/multi_agent/rllib_checkpoints/a2")
            print(f"  [CKPT] Saved checkpoint to {checkpoint_dir}")

        # Check milestones
        if next_milestone_idx < len(a2_milestone_steps) and ts >= a2_milestone_steps[next_milestone_idx]:
            m_str = f"{a2_milestones[next_milestone_idx]}M"
            pol_name = f"a2_snap_{m_str}"
            
            # Copy weights from learning_agent to the snapshot policy
            state = algo.get_policy("learning_agent").get_state()
            algo.get_policy(pol_name).set_state(state)
            
            # sync to workers
            algo.workers.sync_weights()
            print(f"*** Reached {m_str} steps! Snapshot {pol_name} populated with current weights. ***")
            
            next_milestone_idx += 1

if __name__ == "__main__":
    main()
