from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Union

import gym
import numpy as np

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.util import obs_space_info

class DummyRvecEnv(DummyVecEnv):
    """
    In order to be used by RL methods that require a vectorized environment,
    but where a single environment is used to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], agent):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_spaces[agent]
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = env.metadata
    
    def step_wait(self):
        for env_idx in range(self.num_envs):
            obs, rews, dones, infos, ag, next_ag = self.envs[env_idx].step(self.actions[env_idx])
            if dones:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                # print(self.buf_infos)
                obs = self.envs[env_idx].reset()
        return (obs, rews, dones, infos, ag, next_ag)
    
    # def step_wait(self):
    #     for env_idx in range(self.num_envs):
    #         obs, rews, dones, infos, ag, next_ag = self.envs[env_idx].step(self.actions[env_idx])
    #     return (obs, rews, dones, infos, ag, next_ag)
    
    def step_proceed(self, obs, rews, dones, infos):
        for env_idx in range(self.num_envs):
            self._save_obs(env_idx, obs)
            self.buf_rews[env_idx] = rews
            self.buf_dones[env_idx] = dones
            self.buf_infos[env_idx] = infos

    def get_obs(self):
        return self.buf_infos[0]["terminal_observation"] 

    def reset(self):
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
        return obs
