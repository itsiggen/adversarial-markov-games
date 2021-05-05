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
    
    arch = trial.suggest_categorical('architecture', [32,64])
    buffer = trial.suggest_categorical('buffer', [256,512,1024])
    batch = trial.suggest_categorical('batch', [32,64,128])
    # sde = trial.suggest_categorical('sde', [False, True])
    lr = trial.suggest_float('lr',0.00001, 0.001, step=0.00001)
    gamma = trial.suggest_float('gamma', 0.8, 0.99, step=0.01)
    ent_coef = trial.suggest_float('ent_coef', 0, 1e-3, step=1e-4)
    reward = trial.suggest_categorical('reward', [1,2,3,4,5])
    
    # Create environment
    env = gym.make("BagsSkip-v0", steps=1000, rewarder=reward, dataset=dataset)
    """
    Best parameters: {'architecture': 32, 'buffer': 1024, 'batch': 64, 'sde': False,
                      'lr': 0.00039, 'gamma': 0.92, 'ent_coef': 0.0001, 'reward': 4}
    Best median_epsilon: 2.76
    """
        
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=buffer,
        batch_size=batch,
        learning_rate=round(lr,5),
        gamma=gamma,
        tensorboard_log=None,
        ent_coef = round(ent_coef,5),
        # use_sde=sde,
        verbose=0,
        seed=2,
        policy_kwargs=dict(net_arch=[dict(vf=[arch,arch], pi=[arch,arch])])
    )
    
    save_dir = 'mods/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.learn(total_timesteps=int(1e6))
    model.save(save_dir + "bagsskip_" + str(trial.number) + "_model.pt")

    # Evaluate the agent
    envv = gym.make("BagsSkip-v0", steps=1000, rewarder=1, dataset=dataset, train=False)#, nonadaptive=True)
    mean_reward, std_reward, epsilons, _ = evaluate_policy(model, envv, n_eval_episodes=100)
    
    # mean_epsilon = np.mean([x[-1] for x in epsilons])
    median_epsilon = np.median([x[-1] for x in epsilons])
    
    return median_epsilon