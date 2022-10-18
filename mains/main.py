import argparse
import gym
import pandas as pd
import numpy as np
import optuna
from tests.mnist_att_opt import objective_bnd
from tests.mnist_bags_opt import objective_bags
from tests.mnist_hsja_opt import objective_hsja
from tests.cifar_bags_opt import objective_cbags
from tests.cifar_hsja_opt import objective_chsja

study = optuna.create_study()  # Create a new study.
study.optimize(objective_hsja, n_trials=1, gc_after_trial=True)  # Invoke optimization of the objective function.
# study.optimize(objective_hsja, n_trials=100) 


# if __name__ == '__mainatt__':
#     parser = argparse.ArgumentParser(description="Train DQN on BoundarySkip")
#     parser.add_argument('--max-timesteps', default=int(5e4), type=int, help="Maximum number of timesteps")
#     args = parser.parse_args()
#     mainatt(args)

