from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch

class HaxballCallbacks(DefaultCallbacks):
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
