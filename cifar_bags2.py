import argparse
import gym
import os
import pandas as pd
import numpy as np
import optuna
import cProfile
from tqdm import tqdm
from agents.rppo import RPPO
from agents.benign import RandomAgent
from torchvision import datasets, transforms
from utils.evaluation import evaluate_rdpolicy, evaluate_rtpolicy
from envs.bags_games_cifar import BagsGamesCIFAR
from stable_baselines3.common.vec_env import VecNormalize
os.environ['CUDA_VISIBLE_DEVICES'] = ''

transform = transforms.ToTensor()
dataset = datasets.CIFAR10('./data', train=False, transform=transform, download=True)

eval_steps = 5000
adaptive = 0 # non-adaptive, just stateful defense 
ratio = 0.5
defended = True
cont = 0
seed = 2

# Make evaluation env
env = gym.make("BagsGamesCIFAR-v0",
               steps=eval_steps,
               ratio_benign=ratio,
               adaptive=adaptive,
               dataset=dataset,
               defended=defended,
               train=False,
               rint=1,
               radv=1,
               intercept=1)
    
interceptor = RPPO.load("mods/games/bags4int_14.pt" , env, "interceptor", seed)
adversary = RPPO.load("mods/games/bags4adv.pt", env, "adversary", seed)
benign = RandomAgent(env=env)

mean_rint, std_rint, mean_radv, std_radv, epsilons, iters, mean_eps, start_eps, mean_acc = evaluate_rtpolicy(interceptor, adversary, benign, env, act_size=4, n_eval_episodes=100)


res = [mean_eps, start_eps, mean_acc]
c = np.mean([i[1000] for i in epsilons])
d = np.mean([i[2000] for i in epsilons])
print('bags2:', res, c, d)