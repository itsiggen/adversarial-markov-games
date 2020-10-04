import numpy as np
import warnings
import random
import gym
import torch
from utils import agent_selector
from gym import spaces

import math


class amg(gym.Env):
    """Custom Environment that follows gym interface
    Represents adaptive control in adversarial attack and defense policies
    Adversary learns optimized attack hyperparameters
    Interceptor learns optimal defensive policies"""
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super().__init__()

        self.adversaries = 1
        self.interceptors = 1
        self.agents = ["adversary", "interceptor"]

        # Actions controlled by the adversary
        self.action_space_adv = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        # Actions controlled by the interceptor
        self.action_space_int = spaces.Discrete(3)
        # The adversary observes the black-box decisions
        self.observation_space_adv = spaces.Discrete(10)
        # The interceptor observes the adversary queries
        self.observation_space_int = spaces.Box(low=0, high=1, shape=(28, 28), dtype=np.float32)

        self.rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def observe(self, agent):
        board_vals = np.array(self.board.squares).reshape(3, 3)
        cur_player = self.agents.index(self.agent_selection)
        opp_player = (cur_player + 1) % 2

        cur_p_board = np.equal(board_vals, cur_player + 1)
        opp_p_board = np.equal(board_vals, opp_player + 1)
        return np.stack([cur_p_board, opp_p_board], axis=2).astype(np.int8)
    
    def reset(self):
        # Reset environment
        # initialize history?
        #self.board = Board()

        self.rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}

        # selects the first agent
        self.agent_selection = self._agent_selector.reset()


    # action in this case is a value from 0 to 8 indicating position to move on tictactoe board
    def step(self, action):

        # Select next agent
        self.agent_selection = self._agent_selector.next()
                
        # ============== OPTIONAL ==================
        # define the actions
        # action_scaled = self.action_scale(action[0])
        self.action_perlin_freq = self.action_scale_perlin_freq(action[0])
        self.action_mask_factor = self.action_scale_mask_factor(action[1])
        self.action_spherical_step_size = self.action_scale_step_size(action[2])
        self.action_source_step_size = self.action_scale_step_size(action[3])

        self._BA_step()

        # check if we exceeded query budget
        done = bool(self.query_current >= self.query_max)
        # done = bool(self.query_current >= self.query_max or self.dist_opt_adv<3)
        self.query_current += 1
        observation = self.gen_observation()

        reward = self.reward(1)

        # Return current agent
        info = self.agent_selection
        
        #if done close tb
        
        step_attack.tb.close()

        return observation, reward, done, info
        
    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as some print statement wich represent some relevant metrics
        if self.query_current % 99 == 0:
            bbh.plot_label_image('adv pred : ', self.y_opt_adv, self.x_opt_adv)
            # print(f"source steps : {self.state_steps_source}")
            # print(f"spherical steps : {self.state_steps_spherical}")
            print(f"current step : {self.query_current}")
            print(f"distant adv : {self.dist_opt_adv}")

        board = list(map(getSymbol, self.board.squares))

        print(" " * 5 + "|" + " " * 5 + "|" + " " * 5)
        print(f"  {board[0]}  " + "|" + f"  {board[3]}  " + "|" + f"  {board[6]}  ")
        print("_" * 5 + "|" + "_" * 5 + "|" + "_" * 5)

        print(" " * 5 + "|" + " " * 5 + "|" + " " * 5)
        print(f"  {board[1]}  " + "|" + f"  {board[4]}  " + "|" + f"  {board[7]}  ")
        print("_" * 5 + "|" + "_" * 5 + "|" + "_" * 5)

        print(" " * 5 + "|" + " " * 5 + "|" + " " * 5)
        print(f"  {board[2]}  " + "|" + f"  {board[5]}  " + "|" + f"  {board[8]}  ")
        print(" " * 5 + "|" + " " * 5 + "|" + " " * 5)

    def close(self):
        pass