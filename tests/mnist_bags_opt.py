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

transform=transforms.ToTensor()
dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)

# Objective function to be minimized.

def objective_bags(trial):
    from envs.bags_skip import BagsSkip
    
    arch = trial.suggest_categorical('architecture', [8,16,32])
    sde = trial.suggest_categorical('sde', [False, True])
    lr = trial.suggest_discrete_uniform('lr', 0.00001, 0.001, 0.00001)
    gamma = trial.suggest_discrete_uniform('gamma', 0.75, 0.99, 0.01)
    ent_coef = trial.suggest_discrete_uniform('ent_coef', 0, 1e-3, 1e-4)
    reward = trial.suggest_categorical('reward', [1,2,3,4,5])
    
    # Create environment
    env = gym.make("BagsSkip-v0", steps=1000, rewarder=reward, dataset=dataset)
    """
    Best parameters: {'architecture': 8, 'sde': False, 'lr': 5.6e-5, 'gamma': 0.89, 'ent_coef': 0, 'reward': 1}
    Best mean_epsilon: 4.14
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
        policy_kwargs=dict(net_arch=[dict(vf=[arch,arch], pi=[arch,arch])])
    )
    model.learn(total_timesteps=int(1e6))

    # Evaluate the agent
    envv = gym.make("BagsSkip-v0", steps=1000, rewarder=reward, dataset=dataset, train=False)#, nonadaptive=True)
    mean_reward, std_reward, epsilons, _ = evaluate_policy(model, envv, n_eval_episodes=100)
    
    # mean_epsilon = np.mean([x[-1] for x in epsilons])
    median_epsilon = np.median([x[-1] for x in epsilons])
    
    return median_epsilon