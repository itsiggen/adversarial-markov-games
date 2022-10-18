import argparse
import gym
import pandas as pd
import numpy as np
import supersuit as ss
from stable_baselines3 import PPO, DDPG
from torchvision import datasets, transforms
from utils.evaluation import evaluate_policy
from envs.boundary_skip import BoundarySkip
from envs.bags_skip import BagsSkip

"""
Train and save the DQN model for the boundary attack env
:param args: (ArgumentParser) the input arguments
"""
transform=transforms.ToTensor()
dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)

# Create environment
env = gym.make("BagsSkip-v0", steps=1000, rewarder=1, dataset=dataset)#, nonadaptive=True)
# env = gym.make("HsjaSkip-v0", steps=5000, rewarder=1, dataset=dataset)#, nonadaptive=True)
# 'architecture': 8, 'sde': False, 'lr': 0.000056, 'gamma': 0.89, 'ent_coef': 0, 'reward': 1}, best mean_epsilon: 4.14

# env = ss.concat_vec_envs_v0(env, 87, num_cpus=4, base_class='stable_baselines3')

model = DDPG(
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
model.learn(total_timesteps=int(1e6))
# model.learn(total_timesteps=int(5e4))

# print("Saving model to hsjaskip_model.zip")
# model.save("hsjaskip_model")

# # Load the trained agent
# model = PPO.load('mods/hsjaskip_best.pt')

# Evaluate the agent
envv = gym.make("BagsSkip-v0", steps=1000, train=False, dataset=dataset)#, nonadaptive=True)
mean_reward, std_reward, epsilons, mean_eps, start_eps, iters, mean_length, _ = evaluate_policy(model, envv, n_eval_episodes=100)

# mean_epsilon = np.mean([x[-1] for x in epsilons])
# median_epsilon = np.median([x[-1] for x in epsilons])

res = np.asarray([mean_reward, std_reward, mean_eps, start_eps, mean_length])
# np.savetxt('./logs/att.csv', res, delimiter=";", fmt='%1.3f')
print(res)

# df = pd.DataFrame({'mean_reward': mean_reward, 'std_reward': std_reward, 'mean_epsilon': mean_epsilon})
# # file_path = os.path.join(logdir, '50benign.csv')
# df.to_csv('/logs/50benign.csv', index=False, float_format='%.3f')

# if __name__ == '__mainatt__':
#     parser = argparse.ArgumentParser(description="Train DQN on BoundarySkip")
#     parser.add_argument('--max-timesteps', default=int(5e4), type=int, help="Maximum number of timesteps")
#     args = parser.parse_args()
#     mainatt(args)
    
