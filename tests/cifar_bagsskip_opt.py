import gym
import random
import pandas as pd
import numpy as np
import os, sys
import optuna
from stable_baselines3 import PPO
# from envs.boundary_skip import BoundarySkip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.evaluation import evaluate_policy


# def objective(trial):
#     from envs.bags_skip_cifar import BagsSkipCIFAR
    
#     arch = trial.suggest_categorical('architecture', [16,32])
#     buffer = trial.suggest_categorical('buffer', [128,256,2048])
#     batch = trial.suggest_categorical('batch', [16,32,64,128])
#     lr = trial.suggest_categorical('lr', [0.001,0.0005,0.0001,0.00005,0.00001])
#     gamma = trial.suggest_float('gamma', 0.8, 0.99, step=0.01)
#     ent_coef = trial.suggest_categorical('ent_coef', [0,0.001,0.0001])
#     scale = trial.suggest_categorical('scale', [2,5,10,20])
#     reward = trial.suggest_categorical('reward', [1,2,3,4,5])
    
#     # Create environment
#     env = gym.make("BagsSkipCIFAR-v0", steps=1000, rewarder=reward, scale=scale)#, nonadaptive=True)
#     """
#     Best parameters: {'architecture': 32, 'buffer': 256, 'batch': 64, 'lr': 5e-05,
#                       'gamma': 0.9, 'ent_coef': 0, 'scale': 5, 'reward': 3}
#     2nd best:       {'architecture': 32, 'buffer': 256, 'batch': 32, 'lr': 5e-05,
#                      'gamma': 0.83, 'ent_coef': 0, 'scale': 5, 'reward': 1}
#     Best mean_epsilon: 1.458
#     2nd best:          1.552
#     """
       
#     model = PPO(
#         policy="MlpPolicy",
#         env=env,
#         n_steps=buffer,
#         batch_size=batch,
#         learning_rate=lr,
#         gamma=round(gamma,2),
#         tensorboard_log=None,
#         ent_coef=ent_coef,
#         # use_sde=sde,
#         verbose=0,
#         seed=2,
#         policy_kwargs=dict(net_arch=[dict(vf=[arch,arch], pi=[arch,arch])])
#     )
    
#     save_dir = '../mods/'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     model.learn(total_timesteps=int(1e6))
#     model.save(save_dir + "bagsskipcifar_" + str(trial.number) + "_model.pt")

#     # Evaluate the agent
#     envv = gym.make("BagsSkipCIFAR-v0", steps=1000, rewarder=reward, train=False)#, nonadaptive=True)
#     model = PPO.load('../mods/bagsskipcifar_' + str(trial.number) + '_model.pt')
#     mean_reward, std_reward, epsilons, mean_eps, start_eps, iters, mean_length, _ = evaluate_policy(model, envv, n_eval_episodes=100)
    
#     # mean_eps = np.mean([x[-1] for x in epsilons])
#     # median_epsilon = np.median([x[-1] for x in epsilons])
        
#     res = np.asarray([mean_reward, std_reward, mean_eps, start_eps, mean_length])
#     # np.savetxt('./logs/50benign.csv', res, delimiter=";", fmt='%1.3f')
#     print(res)
    
#     return mean_eps

# if __name__ == '__main__':
#     # sampler = optuna.samplers.TPESampler(seed=2)
#     study = optuna.create_study(direction="minimize")
#     study.optimize(objective, n_trials=100)
#     # train(args)
    
def train():
    from envs.bags_skip_cifar import BagsSkipCIFAR
    
    arch = 32
    buffer = 256
    batch = 64
    lr = 0.00005
    gamma = 0.9
    ent_coef = 0
    scale = 5
    reward = 3
    
    # # Create environment
    # env = gym.make("BagsSkipCIFAR-v0", steps=10000, rewarder=reward, nonadaptive=True)
    # """
    # Best parameters: {'architecture': 32, 'buffer': 1024, 'batch': 64, 'sde': False,
    #                   'lr': 0.00039, 'gamma': 0.92, 'ent_coef': 0.0001, 'reward': 4}
    # Best median_epsilon: 2.76
    # """
    
    # model = PPO(
    #     policy="MlpPolicy",
    #     env=env,
    #     n_steps=buffer,
    #     batch_size=batch,
    #     learning_rate=round(lr,5),
    #     gamma=gamma,
    #     tensorboard_log=None,
    #     ent_coef=round(ent_coef,5),
    #     # use_sde=sde,
    #     verbose=0,
    #     seed=2,
    #     policy_kwargs=dict(net_arch=[arch,arch])
    # )
    
    # save_dir = '../mods/'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # model.learn(total_timesteps=int(1e5))
    # model.save('../mods/bagsskipcifar_0_model.pt')
    
    # Evaluate the agent
    envv = gym.make("BagsSkipCIFAR-v0", steps=10000, rewarder=reward, train=False)#, nonadaptive=True)
    model = PPO.load('../mods/bagsskipcifar_best.pt')
    mean_reward, std_reward, epsilons, mean_eps, start_eps, iters, mean_length, _ = evaluate_policy(model, envv, n_eval_episodes=100)
    
    # mean_eps = np.mean([x[-1] for x in epsilons])
    # median_epsilon = np.median([x[-1] for x in epsilons])
        
    res = np.asarray([mean_reward, std_reward, mean_eps, start_eps, mean_length])
    # np.savetxt('./logs/50benign.csv', res, delimiter=";", fmt='%1.3f')
    print(res)


if __name__ == '__main__':
    train()