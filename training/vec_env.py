import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn

class HaxballMultiAgentVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        
        self.agents_per_env = []
        for env in self.envs:
            env.unwrapped.multi_agent = True
            obs, _ = env.reset()
            self.agents_per_env.append(len(obs))
            
        num_envs = sum(self.agents_per_env)
        env = self.envs[0]
        super().__init__(num_envs, env.observation_space, env.action_space)
        
        self.buf_obs = np.zeros((self.num_envs,) + self.observation_space.shape, dtype=self.observation_space.dtype)
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        
        self.actions = None

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        idx = 0
        for env_idx, env in enumerate(self.envs):
            n_agents = self.agents_per_env[env_idx]
            env_actions = self.actions[idx : idx + n_agents]
            
            obs_list, reward_list, term_list, trunc_list, info_list = env.step(env_actions)
            
            done = term_list[0] or trunc_list[0]
            if done:
                for i in range(n_agents):
                    info_list[i]["terminal_observation"] = obs_list[i]
                obs_list, _ = env.reset()
                
            for i in range(n_agents):
                self.buf_obs[idx + i] = obs_list[i]
                self.buf_rews[idx + i] = reward_list[i]
                self.buf_dones[idx + i] = done
                self.buf_infos[idx + i] = info_list[i]
                
            idx += n_agents
            
        return self.buf_obs.copy(), self.buf_rews.copy(), self.buf_dones.copy(), self.buf_infos.copy()

    def reset(self) -> np.ndarray:
        idx = 0
        for env_idx, env in enumerate(self.envs):
            n_agents = self.agents_per_env[env_idx]
            obs_list, _ = env.reset()
            for i in range(n_agents):
                self.buf_obs[idx + i] = obs_list[i]
            idx += n_agents
        return self.buf_obs.copy()

    def close(self) -> None:
        for env in self.envs:
            env.close()
        
    def get_attr(self, attr_name, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self.envs[0], attr_name) for _ in indices]

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if method_name == 'render':
            return [env.render(*method_args, **method_kwargs) for env in self.envs]
        return None

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
