import argparse
import gym
import pandas as pd
import numpy as np
import optuna
from tests.mnist_att_opt import objective_bnd
from tests.mnist_bags_opt import objective_bags

study = optuna.create_study()  # Create a new study.
# study.optimize(objective_bnd, n_trials=100)  # Invoke optimization of the objective function.
study.optimize(objective_bags, n_trials=100)  # Invoke optimization of the objective function.


# if __name__ == '__mainatt__':
#     parser = argparse.ArgumentParser(description="Train DQN on BoundarySkip")
#     parser.add_argument('--max-timesteps', default=int(5e4), type=int, help="Maximum number of timesteps")
#     args = parser.parse_args()
#     mainatt(args)
    
