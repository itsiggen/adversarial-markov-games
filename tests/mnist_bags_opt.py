import gym
import numpy as np
import optuna
import os, sys
from torchvision import datasets, transforms
from stable_baselines3 import PPO
from utils.evaluation import evaluate_policy
# from envs.boundary_skip import BoundarySkip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# Objective function to be minimized.

def objective_bags(trial):
    from envs.bags_skip import BagsSkip
    
    arch = trial.suggest_categorical('architecture', [32,64,128])
    buffer = trial.suggest_categorical('buffer', [512,1024,2048])
    batch = trial.suggest_categorical('batch', [32,64,128])
    steps = trial.suggest_categorical('steps', [1000,2000,3000])
    total = trial.suggest_categorical('total', [1e5,1e6,5e6])
    lr = trial.suggest_categorical('lr', [0.005,0.001,0.0005,0.0003,0.0001])
    epochs = trial.suggest_categorical('epochs', [10,20,30])
    gamma = trial.suggest_float('gamma', 0.8, 0.99, step=0.01)
    ent_coef = trial.suggest_categorical('ent_coef', [0,0.001,0.0005,0.0001,0.00005])
    scale = trial.suggest_categorical('scale', [5,10,20,25])
    reward = trial.suggest_categorical('reward', [1,2,3,4])
    
    # Create environment
    env = gym.make("BagsSkip-v0", steps=steps, rewarder=reward, scale=scale)
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
        n_epochs = epochs,
        gamma=round(gamma,2),
        tensorboard_log=None,
        ent_coef = round(ent_coef,5),
        # use_sde=sde,
        verbose=0,
        seed=2,
        policy_kwargs=dict(net_arch=[dict(vf=[arch,arch], pi=[arch,arch])])
    )
    model.learn(total_timesteps=int(total))
    model.save("mods/bagsskip_" + str(trial.number) + "_model.pt")

    # Evaluate the agent
    envv = gym.make("BagsSkip-v0", steps=5000, rewarder=reward, scale=scale, train=False)#, nonadaptive=True)
    model = PPO.load("mods/bagsskip_" + str(trial.number) + "_model.pt")
    mean_reward, std_reward, epsilons, mean_eps, start_eps, iters, mean_length, _ = evaluate_policy(model, envv, n_eval_episodes=100)
    
    res = np.asarray([mean_reward, std_reward, mean_eps, start_eps])
    print(res)
    
    return mean_eps