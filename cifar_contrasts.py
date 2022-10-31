import argparse
import gym
import os
import pandas as pd
import numpy as np
import optuna
from tqdm import tqdm
from agents.rppo import RPPO
from agents.benign import RandomAgent
from torchvision import datasets, transforms
from utils.evaluation import evaluate_rdpolicy
from envs.hsja_games_cifar import HsjaGamesCIFAR
from stable_baselines3.common.vec_env import VecNormalize
os.environ['CUDA_VISIBLE_DEVICES'] = ''

transform = transforms.ToTensor()
dataset = datasets.CIFAR10('./data', train=False, transform=transform, download=True)
    

def check_full(agents):
    for i in range(2):
        # print(agents[i].rollout_buffer.pos)
        if agents[i].rollout_buffer.full:
            # print(i, "agent training")
            agents[i].close_buffer()
            # agents[i].train()
            agents[i].reset_buffer()

def reset():
    return False, 1, 0, 0

    
adaptive = 2 # int adaptive 
ratio = 0.5
defended = True
seed = 2

steps = 1000
lr = 0.001
buffer = 2048
batch = 32
epochs = 20
gamma = 0.99
ent_coef = 0
vf_coef = 0.5
que = True
radv = 1
rint = 5
inter = 1
ts = 2e4

# Create environment
env = gym.make("HsjaGamesCIFAR-v0",
               steps=steps,
               ratio_benign=ratio,
               adaptive=adaptive,
               dataset=dataset,
               train=que,
               rint=rint,
               radv=radv,
               defended=defended,
               intercept=inter)

total_timesteps = int(ts)

interceptor = RPPO(policy="MlpPolicy",
            env=env,
            agent='interceptor',
            n_steps=buffer,
            batch_size=batch,
            n_epochs=epochs,
            learning_rate=lr,
            gamma=round(gamma,2),
            tensorboard_log=None,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            verbose=0,
            seed=seed,
            # policy_kwargs=dict(net_arch=[32,32]))
            policy_kwargs=dict(net_arch=[dict(vf=[32,32], pi=[32,32])]))
 
# print(env.action_space)
adversary = RPPO(policy="MlpPolicy",
            env=env,
            agent='adversary',
            n_steps=buffer,
            learning_rate=lr,
            gamma=round(gamma,2),
            tensorboard_log=None,
            # ent_coef = ent_coef,
            verbose=0,
            seed=seed,
            mode=1,
            policy_kwargs=dict(net_arch=[32,32]))

benign = RandomAgent(env=env)
  
agents = [interceptor, adversary, benign]


for agent in agents:
    agent.setup_learn()
obs = env.reset()
agents[0].set_last(obs, False)
done = False
curr, nxt = 1, 0
n_steps = 0
rst = 1
    
for timestep in tqdm(range(total_timesteps), disable=False):
    # Check if a rollout buffer has been filled and train
    check_full(agents)
    obs, reward, done, info = agents[nxt].move()
    curr = info["curr"]
    nxt = info["next"]
    # print(curr,nxt)
    n_steps += 1
    # print("cadence", prev, curr, nxt, reward)

    if curr == 0:
        if nxt == 0:
            agents[0].proceed(obs, reward, done, info)
        elif nxt == 1:
            if rst == 1:
                # First time adv plays after start of episode, set first obs
                # to what 
                agents[1].set_last(obs, False)
                rst = 0
            else:
                agents[1].proceed(obs, reward, done, info)
                # print(agents[1].rollout_buffer.pos)
        # elif nxt == 2:
            # if ben is next, do not proceed
            # agents[1].proceed(obs, reward, done, info)
    elif curr == 1 or curr == 2:
        if done:
            # term_obs = agents[1].env.get_obs()
            agents[0].proceed(obs, reward, False, info)
            done, curr, nxt, n_steps = reset()
            rst = 1
        else:
            agents[0].proceed(obs, reward, done, info)

# Save the trained agents

env.contrasts.save()

print("Saving contrasts...")
