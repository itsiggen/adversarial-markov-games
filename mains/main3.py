from PPO import PPO, Memory
# from envs.bezcaptcha import bezCaptcha
from envs.bezdistri import bezDistri
import gym
import torch
import numpy as np
import pyautogui as ag
import gc
import os
import time

############## Hyperparameters ##############
env_name = 'beAbstract-v0'

max_episodes = 6            # max training episode
max_timesteps = 25          # max timesteps in one episode

update_timestep = 10       # update policy every n timesteps
action_std = 0.01            # constant std for action distribution (Multivariate Normal)
K_epochs = 80               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.9                 # discount factor

lr = 0.0003                 # parameters for Adam optimizer
betas = (0.9, 0.999)

random_seed = None
#############################################

# creating environment
env = gym.make(env_name, max_req=max_timesteps, patience=10)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

memory = Memory()
ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
resume_previous_training = True

# logging variables
episode_reward = 0
avg_length = 0
time_step = 0

# load old policy
if resume_previous_training and os.path.exists('./PPO_beAbstract.pt'):
    print("Loading previously saved model ... ")
    ppo.policy_old.load_state_dict(torch.load('./PPO_beAbstract.pt'))
    print("Loaded")

# training loop
for i_episode in range(1, max_episodes+1):
    state = env.reset()
    episode_reward = 0
    for t in range(max_timesteps):
        time_step +=1
        # Running policy_old:
        action = ppo.select_action(state, memory)
        state, reward, done, _ = env.step(action)
        
        # Saving reward and is_terminals:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        
        # update if its time
        if time_step % update_timestep == 0:
            ppo.update(memory)
            memory.clear_memory()
            time_step = 0
        episode_reward += reward
        if done:
            break
    
    # save every 2 episodes
    if i_episode % 2 == 0:
        # control learning
        torch.save(ppo.policy.state_dict(), './PPO_beAbstract.pt')
        scores = env.getScores()
        
    episode_reward = round(episode_reward/i_episode, 2)
    
    print('Episode {} \t Avg reward: {}'.format(i_episode, episode_reward))