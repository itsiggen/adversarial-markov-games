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
    
    arch = trial.suggest_categorical('architecture', [32,64])
    buffer = trial.suggest_categorical('buffer', [128,256,512,1024,2048])
    batch = trial.suggest_categorical('batch', [32,64])
    sde = trial.suggest_categorical('sde', [False, True])
    lr = trial.suggest_float('lr', 0.00001, 0.001, step=0.00001)
    gamma = trial.suggest_float('gamma', 0.8, 0.99, step=0.01)
    ent_coef = trial.suggest_float('ent_coef', 0, 1e-2, step=1e-3)
    reward = trial.suggest_categorical('reward', [1,2,3,4,5])
    
    # Create environment
    env = gym.make("HsjaSkip-v0", steps=5000, rewarder=reward, dataset=dataset)
    """
    Best parameters: {'architecture': 32, 'buffer': 128, 'batch': 64, 'sde': False, 'lr': 0.00053, 
                      'gamma': 0.89, 'ent_coef': 0.0001, 'reward': 1}
    Best mean_epsilon: 2.607
    """
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=buffer,
        batch_size=batch,
        learning_rate=lr,
        gamma=gamma,
        tensorboard_log=None,
        ent_coef = ent_coef,
        use_sde=sde,
        verbose=0,
        seed=2,
        policy_kwargs=dict(net_arch=[dict(vf=[arch,arch], pi=[arch,arch])])
    )
    model.learn(total_timesteps=int(5e4))
    model.save("mods/hsjaskip_" + str(trial.number) + "_model")

    # Evaluate the agent
    envv = gym.make("HsjaSkip-v0", steps=5000, rewarder=reward, dataset=dataset, train=False)#, nonadaptive=True)
    mean_reward, std_reward, epsilons, _ = evaluate_policy(model, envv, n_eval_episodes=100)
    
    mean_epsilon = np.mean([x[-1] for x in epsilons])
    median_epsilon = np.median([x[-1] for x in epsilons])
 
    # df = pd.DataFrame({'mean_reward': [mean_reward], 'std_reward': [std_reward], 'mean_epsilon': [mean_epsilon]})
    # # file_path = os.path.join(logdir, '50benign.csv')
    # df.to_csv('./logs/'+str(trial.number)+'.csv', index=False, float_format='%.3f')
    
    return median_epsilon
