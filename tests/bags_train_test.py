import argparse
import gym
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from torchvision import datasets, transforms
from utils.evaluation import evaluate_policy
from envs.boundary_skip import BoundarySkip
from envs.bags_skip import BagsSkip

"""
Train and save the DQN model for the boundary attack env
:param args: (ArgumentParser) the input arguments
"""
transform=transforms.ToTensor()
dataset = datasets.MNIST('./data', train=False, transform=transform)

# Create environment
env = gym.make("BagsSkip-v0", steps=1000, rewarder=1, dataset=dataset)

# # 'architecture': 16, 'sde': False, 'lr': 0.00076, 'gamma': 0.75, 'ent_coef': 0.001, 'reward': 1}, best mean_epsilon: 3.00

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0001,
    gamma=0.90,
    tensorboard_log=None,
    # ent_coef = 0.00005,
    # use_sde=True,
    verbose=0,
    seed=2,
    policy_kwargs=dict(net_arch=[16,16])
    # policy_kwargs=dict(net_arch=[64, dict(vf=[64], pi=[32, 32])])
)
model.learn(total_timesteps=int(1e5))

# print("Saving model to boundaryskip_model.zip")
# model.save("bagsskip_model")

# Load the trained agent
# model = PPO.load("boundaryskip_model")
model = PPO.load("bagsskip_model")

# Evaluate the agent
envv = gym.make("BagsSkip-v0", steps=1000, train=False, dataset=dataset, nonadaptive=True)
mean_reward, std_reward, epsilons, _ = evaluate_policy(model, envv, n_eval_episodes=100)

mean_epsilon = np.mean([x[-1] for x in epsilons])
median_epsilon = np.median([x[-1] for x in epsilons])

res = np.asarray([mean_reward, std_reward, mean_epsilon, median_epsilon])
# np.savetxt('./logs/att.csv', res, delimiter=";", fmt='%1.3f')
print(res)