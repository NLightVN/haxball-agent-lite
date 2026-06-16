from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
import random
import numpy as np

class HaxballCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        # worker-local storage for win rates
        self.win_rates = {}
        self.match_counts = {}

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        # Sync timesteps to env if available in global_vars
        ts = worker.global_vars.get("timesteps_total", 0)
        envs = base_env.get_sub_environments()
        for env in envs:
            if hasattr(env, "set_timesteps"):
                env.set_timesteps(ts)

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        envs = base_env.get_sub_environments()
        env = envs[env_index]
        
        # Determine if learning_agent won
        # scores[0] is RED, scores[1] is BLUE
        # Usually learning_agent is RED. Let's record the score margin.
        margin = env.scores[0] - env.scores[1]
        
        # Save to custom metrics so driver can aggregate
        # Find which policy blue_0 was using
        opp_policy = episode.policy_for("blue_0")
        if opp_policy and opp_policy != "learning_agent":
            # 1 = win, 0 = loss/draw
            win = 1.0 if margin > 0 else 0.0
            episode.custom_metrics[f"vs_{opp_policy}_win"] = win

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        timesteps = result.get("timesteps_total", 0)
        
        # Update global_vars
        update_dict = {"timesteps_total": timesteps, "timestep": timesteps}
        algorithm.workers.local_worker().set_global_vars(update_dict)
        algorithm.workers.foreach_worker(
            lambda w: w.set_global_vars(update_dict)
        )
        
        # We can also update PFSP win rates here if needed
        # Read from custom_metrics and aggregate
        hist = result.get("custom_metrics", {})
        pfsp_updates = {}
        for k, v in hist.items():
            if k.startswith("vs_") and k.endswith("_win_mean"):
                pol = k[3:-9] # remove 'vs_' and '_win_mean'
                pfsp_updates[pol] = v # mean win rate against this policy in this iter
                
        # Broadcast the updated winrates
        if pfsp_updates:
            algorithm.workers.foreach_worker(
                lambda w: w.global_vars.setdefault("pfsp_winrates", {}).update(pfsp_updates)
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
        """
        Intercepts the trajectory for an agent after the episode completes (or at chunk boundaries).
        Because we use batch_mode="complete_episodes", postprocessed_batch contains the FULL episode.
        This allows us to retroactively add rewards to past steps.
        """
        if "infos" in postprocessed_batch:
            infos = postprocessed_batch["infos"]
            # infos is usually a list or numpy array of dicts
            for t, info in enumerate(infos):
                # RLlib might store infos as a list of dicts.
                if isinstance(info, dict) and "investment_credit" in info:
                    credits = info["investment_credit"]
                    if credits is not None:
                        for steps_ago, amount in credits:
                            target_t = t - steps_ago
                            if 0 <= target_t < len(postprocessed_batch[SampleBatch.REWARDS]):
                                # Patch the reward backwards in time
                                postprocessed_batch[SampleBatch.REWARDS][target_t] += float(amount)
