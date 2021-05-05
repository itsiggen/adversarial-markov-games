from typing import Union
import random
from stable_baselines3.common.type_aliases import GymEnv

class RandomAgent:
    """
    Utility agent to facilitate benign draws
    """

    def __init__(self, env: Union[GymEnv, str]):

        self.env = env
            
    def move(self):

        # Check consequences of returning clipped actions instead of actions
        action = random.uniform(0, 0.2)
        new_obs, reward, done, info, agent, next_agent = self.env.step(action)
        
        return new_obs, reward, done, info, agent, next_agent
    
    def predict(self, obs, state = None, deterministic = False):
        action = random.uniform(0, 0.2)
        state = None
        return action, state
        
    def proceed(self, new_obs, rewards, dones, infos):
        pass
    
    def reset_episode(self):
        pass
    
    def close_episode(self):
        pass
    
    def set_last(self):
        pass
    
    def setup_learn(self):
        pass