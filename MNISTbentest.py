import argparse
import gym
import os
import numpy as np
import optuna
import gc
import tracemalloc
from tqdm import tqdm
from agents.rppo import RPPO
from agents.benign import RandomAgent
from torchvision import datasets, transforms
from utils.evaluation import evaluate_rdpolicy
from envs.bags_games import BagsGames
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

transform=transforms.ToTensor()
dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)


eval_steps = 5000
adaptive = 0
vanilla = True
ratio = 0.5
defended = False
seed = 2

# Make evaluation env
envv = gym.make("BagsGames-v0",
                steps=eval_steps,
                ratio_benign=ratio,
                adaptive=adaptive,
                dataset=dataset,
                defended=defended,
                vanilla=vanilla,
                scale=8,
                train=True,
                rint=4,
                radv=1)


interceptor = RPPO.load("mods/games/bags6int_0.pt", envv, "interceptor", seed)
adversary = RPPO.load("mods/games/bags6adv.pt" , envv, "adversary", seed)

benign = RandomAgent(env=envv)


mean_rint, std_rint, mean_radv, std_radv, epsilons, iters, mean_eps, start_eps, mean_acc = evaluate_rdpolicy(interceptor, adversary, benign, envv, n_eval_episodes=100)

res = [mean_eps, start_eps, mean_acc]

z = list(zip(iters,epsilons))
a = [np.interp(1000, i[0], i[1]) for i in z]
b = [np.interp(2000, i[0], i[1]) for i in z]
c = np.mean(a)
d = np.mean(b)
print(res, c, d)