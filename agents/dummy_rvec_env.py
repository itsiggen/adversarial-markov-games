from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Union

import gym
import numpy as np

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

class DummyRvecEnv(DummyVecEnv):
    """
    In order to be used by RL methods that require a vectorized environment,
    but where a single environment is used to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        super(DummyRvecEnv, self).__init__(env_fns)
    
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
