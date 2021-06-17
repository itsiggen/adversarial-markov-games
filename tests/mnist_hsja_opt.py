import gym
import pandas as pd
import numpy as np
import optuna
import os, sys
from torchvision import datasets, transforms
from stable_baselines3 import PPO
from utils.evaluation import evaluate_policy
import os
# from envs.boundary_skip import BoundarySkip
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

transform=transforms.ToTensor()
# path = os.getcwd() + '/data'
dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)

# Objective function to be minimized.

def objective_hsja(trial):
    from envs.hsja_skip import HsjaSkip
    
    arch = trial.suggest_categorical('architecture', [16,32,64])
    buffer = trial.suggest_categorical('buffer', [256,512,1024])
    batch = trial.suggest_categorical('batch', [16,32,64])
    # sde = trial.suggest_categorical('sde', [False, True])
    lr = trial.suggest_categorical('lr', [0.005,0.001,0.0005,0.0001,0.00005,0.00001])
    gamma = trial.suggest_float('gamma', 0.8, 0.99, step=0.01)
    ent_coef = trial.suggest_categorical('ent_coef', [0,0.001,0.0001])
    reward = trial.suggest_categorical('reward', [1,2,3,4,5])
    scale = trial.suggest_categorical('scale', [40,200,400])
    perlin = trial.suggest_categorical('perlin', [0,1])
    
    # Create environment
    env = gym.make("HsjaSkip-v0", steps=5000, rewarder=reward, dataset=dataset, scale=scale, perlin=perlin)
    """
    Best parameters: {'architecture': 32, 'buffer': 128, 'batch': 64, 'sde': False, 'lr': 0.00053, 
                      'gamma': 0.89, 'ent_coef': 0.0001, 'reward': 1}
    Best mean_eps: 2.607
    
    2nd best: {'architecture': 64, 'buffer': 256, 'batch': 16, 'lr': 0.001,
               'gamma': 0.94, 'ent_coef': 0, 'reward': 2, 'scale': 400}
    
    Beast mean_eps:  2.613 
    """
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=buffer,
        batch_size=batch,
        learning_rate=lr,
        gamma=round(gamma,2),
        tensorboard_log=None,
        ent_coef = ent_coef,
        # use_sde=sde,
        verbose=0,
        seed=2,
        policy_kwargs=dict(net_arch=[arch,arch])
        # policy_kwargs=dict(net_arch=[dict(vf=[arch,arch], pi=[arch,arch])])
    )
    model.learn(total_timesteps=2e4)
    model.save("mods/hsjaskip_" + str(trial.number) + "_model.pt")

    # Evaluate the agent
    envv = gym.make("HsjaSkip-v0", steps=5000, rewarder=reward, dataset=dataset, train=False)#, nonadaptive=True)
    model = PPO.load("mods/hsjaskip_" + str(trial.number) + "_model.pt")
    mean_reward, std_reward, epsilons, mean_eps, start_eps, iters, mean_length, _ = evaluate_policy(model, envv, n_eval_episodes=100)
    
    res = [mean_reward, std_reward, mean_eps, start_eps, mean_length]
    print(res)
 
    # df = pd.DataFrame({'mean_reward': [mean_reward], 'std_reward': [std_reward], 'mean_epsilon': [mean_epsilon]})
    # # file_path = os.path.join(logdir, '50benign.csv')
    # df.to_csv('./logs/'+str(trial.number)+'.csv', index=False, float_format='%.3f')
    
    return mean_eps
