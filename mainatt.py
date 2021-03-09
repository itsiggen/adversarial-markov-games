import argparse
import gym
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from torchvision import datasets, transforms
from utils.evaluation import evaluate_policy
from envs.boundary_skip import BoundarySkip

"""
Train and save the DQN model for the boundary attack env
:param args: (ArgumentParser) the input arguments
"""
transform=transforms.ToTensor()
dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)

# Create environment
env = gym.make("BoundarySkip-v0", steps=1000, rewarder=1, dataset=dataset)

#'architecture': 16, 'sde': False, 'lr': 0.00093, 'gamma': 0.9069, 'ent_coef': .00005, 'reward': 1}, best mean_epsilon: 4.2575

# model = PPO(
#     policy="MlpPolicy",
#     env=env,
#     learning_rate=0.00093,
#     gamma=0.9069,
#     tensorboard_log=None,
#     ent_coef = 0.00005,
#     # use_sde=True,
#     verbose=0,
#     seed=2,
#     policy_kwargs=dict(net_arch=[16,16])
#     # policy_kwargs=dict(net_arch=[64, dict(vf=[64], pi=[32, 32])])
# )
# model.learn(total_timesteps=int(5e5))

# print("Saving model to boundaryskip_model.zip")
# model.save("boundaryskip_model")

#Load the trained agent
model = PPO.load("boundaryskip_model")

# Evaluate the agent
envv = gym.make("BoundarySkip-v0", steps=1000, train=False, dataset=dataset, nonadaptive=True)
mean_reward, std_reward, mean_epsilon, std_epsilon, _ = evaluate_policy(model, envv, n_eval_episodes=100)

res = np.asarray([mean_reward, std_reward, mean_epsilon, std_epsilon])
np.savetxt('./logs/att.csv', res, delimiter=";", fmt='%1.3f')
print(res)

# df = pd.DataFrame({'mean_reward': mean_reward, 'std_reward': std_reward, 'mean_epsilon': mean_epsilon})
# # file_path = os.path.join(logdir, '50benign.csv')
# df.to_csv('/logs/50benign.csv', index=False, float_format='%.3f')

# if __name__ == '__mainatt__':
#     parser = argparse.ArgumentParser(description="Train DQN on BoundarySkip")
#     parser.add_argument('--max-timesteps', default=int(5e4), type=int, help="Maximum number of timesteps")
#     args = parser.parse_args()
#     mainatt(args)
    
