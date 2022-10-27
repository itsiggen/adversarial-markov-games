import gym
import pandas as pd
import numpy as np
import optuna
import os, sys
from torchvision import datasets, transforms
from stable_baselines3 import PPO
from utils.evaluation import evaluate_policy
import os

transform=transforms.ToTensor()
dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)

def objective_hsja(trial):
    from envs.hsja_skip import HsjaSkip
    
    arch = trial.suggest_categorical('architecture', [32,64,128])
    buffer = trial.suggest_categorical('buffer', [128,512,2048])
    batch = trial.suggest_categorical('batch', [32,64,128])
    # sde = trial.suggest_categorical('sde', [False, True])
    lr = trial.suggest_categorical('lr', [0.005,0.001,0.0005,0.0003,0.0001])
    # lr = trial.suggest_float('lr', 0.00001, 0.005, step=0.00001)
    epochs = trial.suggest_categorical('epochs', [10,20,30])
    gamma = trial.suggest_float('gamma', 0.8, 0.99, step=0.01)
    ent_coef = trial.suggest_categorical('ent_coef', [0,0.001,0.0005,0.0001,0.00005])
    reward = trial.suggest_categorical('reward', [1,2,3,4,5])
    # scale = trial.suggest_categorical('scale', [100,200,300,400])
    
    # Create environment
    env = gym.make("HsjaSkip-v0", steps=5000, dataset=dataset, rewarder=reward)
    
    """
    Best 2.567 with 8state and :{'architecture': 64, 'buffer': 512, 'batch': 128, 'lr': 0.0003, 'epochs': 30, 'gamma': 0.99, 'ent_coef': 0.0001, 'reward': 2}.
    
    Best2 2.569 with 5state and: {'architecture': 64, 'buffer': 512, 'batch': 128, 'lr': 0.001, 'epochs': 30, 'gamma': 0.99, 'ent_coef': 0.0005, 'reward': 1}.

    2.579 with 1state and {'architecture': 128, 'buffer': 512, 'batch': 64, 'lr': 0.001, 'gamma': 0.99, 'ent_coef': 5e-05, 'reward': 4}.
    
    Best parameters: {'architecture': 64, 'buffer': 1024, 'batch': 32, 'lr': 0.001,
                'gamma': 0.8, 'ent_coef': 0.0001, 'reward': 1, 'scale': 200}
    Best mean_eps: 2.620
    
    """
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=buffer,
        batch_size=batch,
        learning_rate=round(lr,5),
        n_epochs = epochs,
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
    envv = gym.make("HsjaSkip-v0", steps=5000, rewarder=reward, train=False)#, nonadaptive=True)
    model = PPO.load("mods/hsjaskip_" + str(trial.number) + "_model.pt")
    mean_reward, std_reward, epsilons, mean_eps, start_eps, iters, mean_length, _ = evaluate_policy(model, envv, n_eval_episodes=100)
    
    res = [mean_reward, std_reward, mean_eps, start_eps, mean_length]
    print(res)
 
    # df = pd.DataFrame({'mean_reward': [mean_reward], 'std_reward': [std_reward], 'mean_epsilon': [mean_epsilon]})
    # # file_path = os.path.join(logdir, '50benign.csv')
    # df.to_csv('./logs/'+str(trial.number)+'.csv', index=False, float_format='%.3f')
    
    return mean_eps