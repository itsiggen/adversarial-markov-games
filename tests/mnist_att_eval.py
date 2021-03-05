import gym
import pandas as pd
import numpy as np
import ray
from stable_baselines3 import PPO
from .utils.evaluation import evaluate_policy
from envs.boundary_skip import BoundarySkip
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch

"""
Train and save the DQN model for the boundary attack env
:param args: (ArgumentParser) the input arguments
"""
logdir = "../logs/tb"

def objective(config):
    # Create environment
    env = gym.make("BoundarySkip-v0", steps=1000)
    
    if config["choice"] == "16":
        arch = [16,16]
    elif config["choice"] == "32":
        arch = [32,32]
    else:
        arch = [64,64]
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config["lr"],
        gamma=config["gamma"],
        tensorboard_log=None,
        ent_coef = config["ent_coef"],
        use_sde=config["sde"],
        verbose=0,
        seed=2,
        policy_kwargs=dict(net_arch=arch)
        # policy_kwargs=dict(net_arch=[64, dict(vf=[64], pi=[32, 32])])
    )
    model.learn(total_timesteps=int(5e5))
    
    # print("Saving model to boundaryskip_model.zip")
    # model.save("boundaryskip_model")
    
    # #Load the trained agent
    # model = PPO.load("boundaryskip_model")
    
    # Evaluate the agent
    envv = gym.make("BoundarySkip-v0", steps=1000, train=False)#, nonadaptive=True)
    mean_reward, std_reward, mean_epsilon, _ = evaluate_policy(model, envv, n_eval_episodes=100)
    
    tune.report(mean_loss=mean_epsilon)

def optimize():
    ray.init(configure_logging=False, ignore_reinit_error=True)
    
    # Optional params to initialize the search 
    current_best_params = [{"lr": 1e-4, "gamma": 0.99, "sde": False, "ent_coef": 0, "reward": 2, "arch": "64"}]

    algo = HyperOptSearch(points_to_evaluate=current_best_params)
    algo = ConcurrencyLimiter(algo, max_concurrent=4)

    analysis = tune.run(
        objective,
        search_alg=algo,
        metric="mean_loss",
        mode="min",
        num_samples=1000,
        config={
            "lr": tune.uniform(1e-5, 1e-3),
            "gamma": tune.uniform(0.9, 0.99),
            # Choice params are ignored by hyperopt
            "sde": tune.choice([False, True]),
            "ent_coef": tune.choice([0, 0.001]),
            "reward": tune.choice([1, 2, 3, 4 ,5]),
            "arch": tune.choice(["16", "32", "64"])
        })

    print("Best hyperparameters found were: ", analysis.best_config)
    df = analysis.results_df
    return df

analysis = optimize()