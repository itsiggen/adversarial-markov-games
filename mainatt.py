import argparse
import gym
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from utils.evaluation import evaluate_policy
from envs.boundary_skip import BoundarySkip

"""
Train and save the DQN model for the boundary attack env
:param args: (ArgumentParser) the input arguments
"""
logdir = "./logs/tb"

# Create environment
env = gym.make("BoundarySkip-v0", steps=1000)

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    gamma=0.99,
    tensorboard_log=None,
    # ent_coef = 0.001,
    # use_sde=True,
    verbose=0,
    seed=2,
    policy_kwargs=dict(net_arch=[64,64])
    # policy_kwargs=dict(net_arch=[64, dict(vf=[64], pi=[32, 32])])
)
model.learn(total_timesteps=int(5e4))

print("Saving model to boundaryskip_model.zip")
model.save("boundaryskip_model")

#Load the trained agent
model = PPO.load("boundaryskip_model")

# Evaluate the agent
envv = gym.make("BoundarySkip-v0", steps=1000, train=False, tensorboard = logdir)#, nonadaptive=True)
mean_reward, std_reward, mean_epsilon, _ = evaluate_policy(model, envv, n_eval_episodes=10)

res = np.asarray([mean_reward, std_reward, mean_epsilon])
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
    
