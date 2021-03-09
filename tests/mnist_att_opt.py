import gym
import pandas as pd
import numpy as np
import optuna
import os, sys
from torchvision import datasets, transforms
from stable_baselines3 import PPO
from utils.evaluation import evaluate_policy
# from envs.boundary_skip import BoundarySkip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
"""
Train and save the DQN model for the boundary attack env
:param args: (ArgumentParser) the input arguments
"""

transform=transforms.ToTensor()
dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)

# Objective function to be minimized.

def objective(trial):
    from envs.boundary_skip import BoundarySkip
    
    arch = trial.suggest_categorical('architecture', [[16,16], [32,32], [64,64]])
    sde = trial.suggest_categorical('sde', [False, True])
    lr = trial.suggest_float('lr', 1e-5, 1e-3)
    gamma = trial.suggest_float('gamme', 0.9, 0.99)
    ent_coef = trial.suggest_float('ent_coef', 0, 0.001)
    reward = trial.suggest_categorical('reward', [1,2,3,4,5])
    
    # Create environment
    env = gym.make("BoundarySkip-v0", steps=1000, rewarder=reward, dataset=dataset)
    """
    Best parameters: {'architecture': 16, 'sde': False, 'lr': 0.00093, 'gamma': 0.9069, 'ent_coef': .00005, 'reward': 1}
    Best mean_epsilon: 4.257562637329102.
    """
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr,
        gamma=gamma,
        tensorboard_log=None,
        ent_coef = ent_coef,
        use_sde=sde,
        verbose=0,
        seed=2,
        policy_kwargs=dict(net_arch=arch)
    )
    model.learn(total_timesteps=int(5e5))

    # Evaluate the agent
    envv = gym.make("BoundarySkip-v0", steps=1000, train=False)#, nonadaptive=True)
    mean_reward, std_reward, mean_epsilon, _ = evaluate_policy(model, envv, n_eval_episodes=100)
    
    return mean_epsilon
