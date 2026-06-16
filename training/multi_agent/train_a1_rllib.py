import os
import sys
import time
import numpy as np

# Prevent TensorFlow DLL issues on Windows - must be set before any ray import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'

# Redirect stderr briefly to suppress TF deprecation spam at startup
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import ray
from ray.tune.registry import register_env
# Import directly from submodule to bypass ray.rllib.algorithms.__init__ which
# tries to import CQL -> TensorFlow -> DLL crash on Windows
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.policy import PolicySpec

from training.multi_agent.env_a1_rllib import HaxballA1RLlibEnv

def env_creator(env_config):
    return HaxballA1RLlibEnv(env_config)

class A1CurriculumCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        # Read global step from worker global vars
        global_step = worker.global_vars.get("global_step", 0)
        env = base_env.get_sub_environments()[env_index]
        
        # Always use small map for 1v1
        env.map_type = "small"
        
        # Still use basic bots 50% of the time during warmup
        if global_step < 500_000:
            env.bot_mode = np.random.rand() < 0.5
            if env.bot_mode:
                env.bot_type = np.random.choice(["Static", "Random", "Wanderer", "Pazzo"])
        else:
            env.bot_mode = False

    def on_train_result(self, *, algorithm, result, **kwargs):
        # Sync global step to all workers
        global_step = result["timesteps_total"]
        algorithm.workers.foreach_worker(
            lambda w: w.set_global_vars({
                "global_step": global_step,
                "timestep": global_step,  # Required by RLlib's on_global_var_update internally
            })
        )

    def on_postprocess_trajectory(
        self,
        *,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs,
    ):
        from ray.rllib.policy.sample_batch import SampleBatch
        if "infos" in postprocessed_batch:
            infos = postprocessed_batch["infos"]
            for t, info in enumerate(infos):
                if isinstance(info, dict) and "investment_credit" in info:
                    credits = info["investment_credit"]
                    if credits is not None:
                        for steps_ago, amount in credits:
                            target_t = t - steps_ago
                            if 0 <= target_t < len(postprocessed_batch[SampleBatch.REWARDS]):
                                postprocessed_batch[SampleBatch.REWARDS][target_t] += float(amount)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == "red_0":
        return "learning_agent"
    else:
        if np.random.rand() < 0.5:
            return "old_a1"
        else:
            # Randomly select a snapshot or the current learning agent
            valid_policies = [p for p in worker.policy_map.keys() if p.startswith("snapshot_") or p == "learning_agent"]
            return np.random.choice(valid_policies)

def main():
    ray.init(
        runtime_env={"env_vars": {
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "TF_ENABLE_ONEDNN_OPTS": "0",
        }},
        logging_level="ERROR",  # Suppress Ray verbose logs, but keep ERROR
    )
    
    register_env("haxball_a1_v0", env_creator)
    
    # Check if migrated model exists
    migrated_path = "models/multi_agent/rllib_checkpoints/migrated_a1_21000000"
    if not os.path.exists(migrated_path):
        print(f"ERROR: Old A1 snapshot not found at {migrated_path}")
        return
        
    print("Loading weights from migrated A1...")
    # Register the env name expected by the checkpoint (it was likely haxball_multi_v0)
    from training.multi_agent.env_rllib import HaxballRLlibEnv
    register_env("haxball_multi_v0", lambda c: HaxballRLlibEnv(c))
    
    temp_algo = Algorithm.from_checkpoint(migrated_path)
    
    # Auto-detect policy name (checkpoint may not use "learning_agent")
    available_policies = list(temp_algo.workers.local_worker().policy_map.keys())
    print(f"  Found policies in checkpoint: {available_policies}")
    policy_key = "learning_agent" if "learning_agent" in available_policies else available_policies[0]
    print(f"  Using policy: '{policy_key}'")
    
    old_weights = temp_algo.get_policy(policy_key).get_weights()
    temp_algo.stop()
    print("Old weights successfully loaded.")

    config = (
        PPOConfig()
        .environment("haxball_a1_v0")
        .framework("torch")
        .rollouts(
            num_rollout_workers=2,
            batch_mode="complete_episodes",
        )
        .training(
            train_batch_size=8000,
            sgd_minibatch_size=1000,
            num_sgd_iter=10,
            lr=1e-4, # Lower LR for fine-tuning
            model={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "tanh",
                "vf_share_layers": False,
            }
        )
        .multi_agent(
            policies={
                "learning_agent": PolicySpec(),
                "old_a1": PolicySpec()
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["learning_agent"],
        )
        .callbacks(A1CurriculumCallbacks)
        .update_from_dict({
            "extra_python_environs_for_worker": {
                "RLLIB_TEST_NO_TF_IMPORT": "1"
            }
        })
    )
    
    print("Building RLlib PPO Algorithm for A1 Fine-tuning...")
    algo = config.build()
    
    # Initialize learning_agent and old_a1 with the migrated weights
    algo.get_policy("learning_agent").set_weights(old_weights)
    algo.get_policy("old_a1").set_weights(old_weights)
    
    # Sync workers with the weights (since we manually updated the local algo)
    algo.workers.sync_weights()
    
    os.makedirs("models/multi_agent/rllib_checkpoints/a1_finetune", exist_ok=True)
    
    snapshots_taken = set()
    snapshot_targets = [500_000, 1_000_000, 2_000_000, 4_000_000, 6_000_000, 8_000_000]
    
    print("Starting fine-tuning loop...")
    print(f"{'Iter':>6} | {'Total Steps':>12} | {'Phase':>15} | {'Rew/ep':>8} | {'Len/ep':>8} | {'#Episodes':>10} | {'Time/iter':>10}")
    print("-" * 90)
    t_last = time.time()
    
    for i in range(100000):
        result = algo.train()
        t_now = time.time()
        dt = t_now - t_last
        t_last = t_now
        
        ts = result["timesteps_total"]
        # Try both RLlib 2.x keys
        env_r = result.get("env_runners", result)
        rew_mean = env_r.get("episode_reward_mean", result.get("episode_reward_mean", float('nan')))
        len_mean = env_r.get("episode_len_mean", result.get("episode_len_mean", float('nan')))
        n_ep = env_r.get("episodes_this_iter", result.get("episodes_this_iter", 0))
        
        phase = "Warmup(Bot)" if ts < 500_000 else "SelfPlay"
        rew_str = f"{rew_mean:8.2f}" if rew_mean == rew_mean else "       -"
        len_str = f"{len_mean:8.1f}" if len_mean == len_mean else "       -"
        print(f"{i:6d} | {ts:>12,} | {phase:>15} | {rew_str} | {len_str} | {n_ep:>10} | {dt:8.1f}s")
              
        if i % 10 == 0:
            checkpoint_dir = algo.save("models/multi_agent/rllib_checkpoints/a1_finetune")
            print(f"  [CKPT] Saved to {checkpoint_dir}")
            
        # Check Snapshot logic
        for target in snapshot_targets:
            if ts >= target and target not in snapshots_taken:
                snapshots_taken.add(target)
                snap_name = f"snapshot_{target}"
                
                print(f"*** Reached {target} steps! Creating Self-Play Snapshot: {snap_name} ***")
                
                # Create a new policy dynamically and set its weights
                current_weights = algo.get_policy("learning_agent").get_weights()
                
                # RLlib allows adding policies on the fly
                # Note: snapshot policies are NOT added to policies_to_train (only "learning_agent" is trained)
                algo.add_policy(
                    policy_id=snap_name,
                    policy_cls=type(algo.get_policy("learning_agent")),
                    observation_space=algo.get_policy("learning_agent").observation_space,
                    action_space=algo.get_policy("learning_agent").action_space,
                    config={"model": {"fcnet_hiddens": [64, 64], "fcnet_activation": "tanh", "vf_share_layers": False}},
                    policy_state=algo.get_policy("learning_agent").get_state(),  # Copies full state including weights
                )
                
                # Sync workers to ensure they have the new policy
                algo.workers.sync_weights()
                
                checkpoint_dir = algo.save(f"models/multi_agent/rllib_checkpoints/a1_finetune_snapshot_{target}")
                print(f"*** Saved snapshot checkpoint to {checkpoint_dir} ***")

if __name__ == "__main__":
    main()
