import argparse
import gym
import pandas as pd
import numpy as np
from stable_baselines3 import DQN, PPO
from utils.evaluation import evaluate_policy
from envs.boundary_step import BoundaryStep

def main(args):
    """
    Train and save the DQN model for the boundary attack env
    :param args: (ArgumentParser) the input arguments
    """
    # Create environment
    env = gym.make("BoundaryStep-v0", steps=1001, ratio_benign=0.5)
    logdir = "./logs"
    
    # using layer norm policy here is important for parameter space noise!
    # model = DQN(
    #     policy="MlpPolicy",
    #     env=env,
    #     learning_rate=1e-3,
    #     buffer_size=1000000,
    #     learning_starts=200,
    #     batch_size=32,
    #     gamma=0.99,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.05,
    #     tensorboard_log=None,
    #     verbose=0,
    #     seed=2,
    #     policy_kwargs=dict(net_arch=[64, 64])
    # )
    # model = PPO(
    #     policy="MlpPolicy",
    #     env=env,
    #     learning_rate=1e-4,
    #     gamma=0.9,
    #     use_sde=True,
    #     tensorboard_log=None,
    #     verbose=0,
    #     seed=2,
    #     policy_kwargs=dict(net_arch=[64, 64])
    # )
    # model.learn(total_timesteps=args.max_timesteps)

    # print("Saving model to boundarystep_model.zip")
    # model.save("boundarystep_model")
    
    # Load the trained agent
    model = PPO.load("boundarystep_model")

    # Evaluate the agent
    envv = gym.make("BoundaryStep-v0", steps=1001, ratio_benign=0.5, train=False)
    mean_reward, std_reward, mean_epsilon, mean_acc = evaluate_policy(model, envv, n_eval_episodes=2)
    
    res = np.asarray([mean_reward, std_reward, mean_epsilon, mean_acc])
    np.savetxt('./logs/50benign.csv', res, delimiter=";", fmt='%1.3f')
    print(res)
    
    # df = pd.DataFrame({'mean_reward': mean_reward, 'std_reward': std_reward, 'mean_epsilon': mean_epsilon, 'mean_acc': mean_acc})
    # # file_path = os.path.join(logdir, '50benign.csv')
    # df.to_csv('/logs/50benign.csv', index=Falsef, loat_format='%.3f')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DQN on BoundaryStep")
    parser.add_argument('--max-timesteps', default=int(5e4), type=int, help="Maximum number of timesteps")
    args = parser.parse_args()
    main(args)
    

        
        # self.startTrainingQuery = 200
        
        # def __init__(self, policy, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.1,
        #          exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, batch_size=32, double_q=True,
        #          learning_starts=1000, target_network_update_freq=500, prioritized_replay=False,
        #          prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
        #          prioritized_replay_eps=1e-6, param_noise=False,
        #          n_cpu_tf_sess=None, verbose=0, tensorboard_log=None,
        #          _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None):