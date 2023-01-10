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

    
adaptive = 3
ratio = 0.5
defended = False
cont = 1
seed = 2
scale = 20

steps = 1000
lr = 0.001
buffer = 2048
batch = 64
epochs = 20
gamma = 0.99
radv = 5
rint = 5
ts = 1e5

# Create environment
env = gym.make("BagsGamesCIFAR-v0",
                steps=steps,
                ratio_benign=ratio,
                adaptive=adaptive,
                dataset=dataset,
                scale=scale,
                cont=cont,
                rint=rint,
                radv=radv,
                defended=defended,
                )

total_timesteps = int(ts)

interceptor = RPPO.load("mods/games/cbags4int_2.pt" , env, "interceptor", seed)
adversary = RPPO.load("mods/games/cbags5adv_29.pt", env, "adversary", seed)
benign = RandomAgent(env=env)
  
agents = [interceptor, adversary, benign]


for agent in agents:
    agent.setup_learn()
obs = env.reset()
agents[0].set_last(obs, False)
done = False
curr, nxt = 1, 0
n_steps = 0
    
for timestep in tqdm(range(total_timesteps), disable=False):
    # Check if a rollout buffer has been filled and train
    check_full(agents, stt)
    # Store previous move
    prev = curr
    # next agent moves
    # print(nxt)
    # obs, reward, done, info, curr, nxt = agents[nxt].move()
    obs, reward, done, info = agents[nxt].move()
    curr = info["curr"]
    nxt = info["next"]
    n_steps += 1
    
    if curr == 0:
        if n_steps == 1:
            # env has been just reset
            agents[1].set_last(obs, False)
        else:
            agents[prev].proceed(obs, reward, done, info)
    elif curr == 1 or curr == 2:
        if done:
            # term_obs = agents[1].env.get_obs()
            agents[0].proceed(obs, reward, False, info)
            # print(info['gap'], info['epsilon'], info['correct'])
            # agents[0].set_last(obs, False)
            done, curr, nxt, n_steps = reset()
        else:
            agents[0].proceed(obs, reward, done, info)

# Save the trained agents

env.contrasts.save()

print("Saving contrasts...")
