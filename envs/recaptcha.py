import gym
import os
from math import sin, cos, radians, pi, atan2, degrees, sqrt
from enum import IntEnum
import numpy as np
import pandas as pd
from random import randrange, choice, uniform
from gym import error, spaces, utils
from gym.utils import seeding
import pyautogui as ag
from gym.envs.registration import register
import requests
import datetime
import time
import gc

# Global environment definitions

FPS = 50
ag.PAUSE = 0.01
file_dir = os.getcwd()
csv_folder = 'storage'
today = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
ag.FAILSAFE = False

class reCaptchaEnv(gym.Env):
    
    def __init__(self, agent_pos=None, goal_pos=None, goal_size=None, max_steps=100, step_size = 15, interval = 180, experiment = 0):

        self.agent_pos = agent_pos
        self.goal_pos = goal_pos
        self.goal_size = goal_size
        self.action_space = spaces.Box(np.array([0,-1]), np.array([+1,+1]), dtype=np.float32)  # distance, angle
        self.max_steps = max_steps
        self.step_size = step_size
        self.last_score_a = 0
        self.last_score_b = 0
        self.trajectory_dir = -1
        self.episode_inc = -1
        self.interval = interval
        self.experiment = experiment
        self.scores = []
        self.scores_inc = []
        self.scores_vis = []  
        self.time_inc = []
        self.time_vis = []
        
        # Dataframes for scores & timestamps
        column_names = ["score", "time", "date"]
        self.df1 = pd.DataFrame(columns = column_names)
        self.df2 = pd.DataFrame(columns = column_names)
        
        self.observation_space = spaces.Box(
            np.array([-1000,-1000,-1000,-1000,0]),
            np.array([1000,1000,1000,1000,1000]),
            dtype='uint8')
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })
        # obs = [initx, inity, currx, curry, idx, idy, adx, ady, self.goal_size, int(self.dist)]
        # does the embedding normalize? check if it is required here
        
    def reset(self):
        # gc.collect()
        # Set direction towards to / away from trigger butten
        self.trajectory_dir *= -1
        # Uniform delay
        time.sleep(uniform(0.5, 3))
        # Set goal state
        self.episode_inc += 1
        # Differentiate goal_state between single and double experiment
        self.goal_state = self.episode_inc % 2 if self.experiment else self.episode_inc % 4
        # Current position and direction of the agent
        self.init_pos = ag.position()
        self.agent_pos = self.init_pos
        self.goal_pos, self.goal_size = self.getGoal()
        self.init_dir = self.posToAngle(self.goal_pos, self.agent_pos)
        self.agent_dir = self.init_dir
        self.dist = self.posToDist(self.agent_pos, self.goal_pos)
        self.lasts = None
        self.speed = 0
        
        self.reward = 0.0
        self.prev_reward = 0.0
        self.t = 0.0
        
        # Control the time delay between each query
        if self.episode_inc != 0:
            if self.goal_state == 0:
                time.sleep(uniform(self.interval*0.75, self.interval*1.25))
            elif self.goal_state == 2:
                time.sleep(uniform(self.interval*0.05, self.interval*0.1))
        
        # These fields should have been defined by the start of the episode
        assert self.agent_pos is not None
        assert self.goal_pos is not None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs
    
    def step(self, action):
        self.step_count += 1
        self.t += 1.0/FPS
        done = False
        
        # Convert to relative angle and distance
        distance = max(action[0]*self.dist/self.step_size, 2)
        angle = (action[1]*25*self.trajectory_dir + self.agent_dir) % 360
        
        # Move to new position
        self.agent_pos = self.angleToPos(self.agent_pos, distance, angle)
        # print([self.agent_pos[0], self.agent_pos[1]])
        ag.moveTo(self.agent_pos[0], self.agent_pos[1])
        
        # Calculate the new direction and speed
        self.agent_dir = self.posToAngle(self.agent_pos, self.goal_pos)
        self.speed = distance
            
        # Generate new observation/state
        obs = self.gen_obs()
        
        if self.goalBox():
            if self.trajectory_dir == 1:
                ag.press('esc')
                time.sleep(uniform(0.2, 0.5))
                ag.mouseDown()
                time.sleep(uniform(0.08, 0.12))
                ag.mouseUp()
                time.sleep(uniform(0.3, 0.5))
                try:
                    r = requests.get("http://localhost:5000/alterego/result")
                    if r.json()["challenge_ts"] == self.lasts:
                        self.result = 0
                    else:
                        self.result = r.json()["score"]
                        self.lasts = r.json()["challenge_ts"]
                except:
                    print("Server returned no score")
                    self.result = 0
                self.scores.append(self.result)
                self.reward += self.gen_reward()
            else:
                time.sleep(uniform(0.2, 0.5))
                if self.experiment == 0:
                    ag.click()
                time.sleep(uniform(0.2, 0.5))
                # time.sleep(uniform(0.5, 1))
                ag.press('esc')
                ag.press('f5')
                self.reward += 0
            # print('GOAL')
            done = True
        # if self.step_count >= self.max_steps:
        #     done = True
            
        # Add penalty on steps taken
        self.reward -= 0.001
        step_reward = self.reward - self.prev_reward
        self.prev_reward = self.reward
    
        return obs, step_reward, done, {}
    
    def posToDist(self, p0, p1):
        dx = (p0[0]-p1[0])**2
        dy = (p0[1]-p1[1])**2
        return sqrt(dx + dy)
    
    def posToAngle(self, p0, p1):
        angle = degrees(atan2(p1[1] - p0[1], p1[0] - p0[0]))
        return angle % 360
            
    def angleToPos(self, p0, distance, angle):
        theta = radians(int(angle))
        point = [int(p0[0] + distance * cos(theta)), int(p0[1] + distance * sin(theta))]
        return point
        
    def goalBox(self):
        a = self.agent_pos[0] >= self.goal_pos[0]-self.goal_size and self.agent_pos[0] <= self.goal_pos[0]+self.goal_size
        b = self.agent_pos[1] >= self.goal_pos[1]-self.goal_size and self.agent_pos[1] <= self.goal_pos[1]+self.goal_size
        return a and b
        
    
    # Consider normalization 0-1
    def gen_obs(self):
        initx = self.goal_pos[0] - self.init_pos[0]
        inity = self.goal_pos[1] - self.init_pos[1]
        currx = self.goal_pos[0] - self.agent_pos[0]
        curry = self.goal_pos[1] - self.agent_pos[1]
        # ind = self.init_dir
        # agd = self.agent_dir
        # obs = np.array([initx, inity, currx, curry, ind, agd, self.goal_size, int(self.dist)])
        # obs = np.array([initx, inity, currx, curry, int(self.speed)])
        obs = np.array([initx, inity, currx, curry, int(self.dist)])
        return obs
    
    def gen_reward(self):
        if self.experiment == 0:
            if self.goal_state == 0:
                if self.episode_inc == 0:
                    self.last_score_a = self.result
                    reward = 0
                    self.scores_inc.append(self.result)
                    self.time_inc.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    reward = self.result - self.last_score_a
                    if self.result >= 0.7: reward += 0.01
                    self.last_score_a = self.result
                    self.scores_inc.append(self.result)
                    self.time_inc.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            else:
                if self.episode_inc == 1:
                    self.last_score_b = self.result
                    reward = 0
                    self.scores_vis.append(self.result)
                    self.time_vis.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    reward = self.result - self.last_score_b
                    if self.result >= 0.7: reward += 0.01
                    self.last_score_b = self.result
                    self.scores_vis.append(self.result)
                    self.time_vis.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            return  reward                
        elif self.experiment == 1:
            if self.episode_inc == 0:
                self.last_score_a = self.result
                reward = 0
                self.scores_inc.append(self.result)
                self.time_inc.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            else:
                reward = self.result - self.last_score_a
                self.last_score_a = self.result
                self.scores_inc.append(self.result)
                self.time_inc.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            return  reward
    
    def getGoal(self):
        if self.experiment == 0:
            """
            Returns:
              Bounding box of the goal in coordinates, moving between states.
              1: Trigger button in chrome incognito - zero cookies
              2: Firefox button
              3: Trigger button in firefox with cookies
              4: Chrome button       
        
            """
            if self.goal_state == 0:
                goal = [490, 510]
            elif self.goal_state == 1:
                goal = [135, 1060]            
            elif self.goal_state == 2:
                goal = [490, 510]
            elif self.goal_state == 3:
                goal = [583, 1060]
                
            return goal, 7
        elif self.experiment == 1:
            """
            Returns:
              Bounding box of the goal in coordinates, alternating between
              the trigger button and a random location near the border
        
            """
            if self.trajectory_dir == 1:
                goal = [490, 510]
            else:
                a = choice([(0,300),(700,1000)])
                b = choice([(200,400),(600,800)])
                goal = [randrange(*a), randrange(*b)]
            return goal, 7
    
    def log_results(self):
        if self.experiment == 0:
            df1 = pd.DataFrame({'score': self.scores_inc , 'time': self.time_inc})
            df2 = pd.DataFrame({'score': self.scores_vis , 'time': self.time_vis})
            file_path = os.path.join(file_dir, csv_folder, 'incognito-'+ today +'.csv')
            df1.to_csv(file_path, index=False)
            file_path = os.path.join(file_dir, csv_folder, 'visible-'+ today +'.csv')
            df2.to_csv(file_path, index=False)
        elif self.experiment == 1:
            df1 = pd.DataFrame({'score': self.scores_inc , 'time': self.time_inc})
            file_path = os.path.join(file_dir, csv_folder, 'cognito-'+ today +'.csv')
            df1.to_csv(file_path, index=False)
        else:
            print("wrong exp number")
    
    def getScores(self):
        return(self.scores)

# for env in gym.envs.registry.env_specs:
#     if 'MousePlane-v0' not in env:
#         register(
#             id='MousePlane-v0',
#             entry_point='mouseplane.envs:MousePlaneEnv',
#             reward_threshold=0.95
#             )