import numpy as np
from ray.rllib.policy.policy import Policy

class BotPolicy(Policy):
    """
    A simple heuristic policy to act as Wanderer/Random bots during early 2v2 training.
    """
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self._rng = np.random.default_rng()

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        
        # Action space is MultiDiscrete([9, 2])
        # [dir_idx, kick]
        actions = []
        for _ in range(len(obs_batch)):
            # Random Wanderer: Pick a random direction, rarely kick
            dir_idx = self._rng.integers(0, 9)
            kick = 1 if self._rng.random() < 0.05 else 0
            actions.append([dir_idx, kick])
            
        return actions, [], {}

    def learn_on_batch(self, samples):
        return {}
    
    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass
