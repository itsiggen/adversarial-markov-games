import argparse
import gym
import os
import pandas as pd
import numpy as np
import optuna
from agents.rppo import RPPO
from agents.benign import RandomAgent
from torchvision import datasets, transforms
from utils.evaluation import evaluate_rdpolicy
from envs.bags_games import BagsGames

def objective(trial):
    """
    Train and save the adversary and interceptor agents for BAGS
    :param args: (ArgumentParser) the input arguments
    """
    # timesteps = args.timesteps
    # steps = args.steps
    # adaptive = args.adaptive
    # ratio = args.ratio
    # defended = args.defended
    # seed = args.seed
    # name = args.name
    # logdir = "./logs/"
    
    timesteps = int(4e5)
    steps = 5000
    adaptive = 2
    ratio = 0.5
    defended = False
    seed = 2
    logdir = "./logs/"
    
    lr = trial.suggest_float('lr',0.00001, 0.001, step=0.00001)
    gamma = trial.suggest_float('gamma', 0.9, 0.99, step=0.01)
    ent_coef = trial.suggest_float('ent_coef', 0, 1e-4, step=1e-4)
    inter = trial.suggest_categorical('intercept', [1,2,3,5,8,10])
    reward = trial.suggest_categorical('reward', [1,2,3])
    
    transform=transforms.ToTensor()
    dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)
    
    # Create environment
    env = gym.make("HsjaGames-v0", steps=steps, ratio_benign=ratio, adaptive=adaptive, dataset=dataset, rewarder=reward, defended=defended, intercept = inter, seed=seed)
    total_timesteps = timesteps
    
    interceptor = RPPO(policy="MlpPolicy",
                env=env,
                agent='interceptor',
                n_steps=2048,
                learning_rate=lr, # 0.00039
                gamma=gamma, # 0.92
                tensorboard_log=None,
                ent_coef = ent_coef, # 0.0001
                verbose=0,
                seed=seed,
                policy_kwargs=dict(net_arch=[dict(vf=[32,32], pi=[32,32])]))
     
    adversary = RPPO(policy="MlpPolicy",
                env=env,
                agent='adversary',
                n_steps=128,
                learning_rate=0.00056,
                gamma=0.89,
                tensorboard_log=None,
                # ent_coef = ent_coef,
                verbose=0,
                seed=seed,
                policy_kwargs=dict(net_arch=[8,8]))
    
    benign = RandomAgent(env=env)
      
    agents = [interceptor, adversary, benign]
    
    
    for agent in agents:
        agent.setup_learn()
    obs = env.reset()
    agents[0].set_last(obs, False)
    done = False
    curr, nxt = 1, 0
    n_steps = 0
    rst = 1
        
    for timestep in range(total_timesteps):
        # Check if a rollout buffer has been filled and train
        check_full(agents)
        # Store previous move
        prev = curr
        # next agent moves
        # print(nxt)
        obs, reward, done, info, curr, nxt = agents[nxt].move()
        n_steps += 1

        if curr == 0:
            # if n_steps == 1:
            #     # env has been just reset
            #     agents[1].set_last(obs, False)
            # else:
            #     agents[prev].proceed(obs, reward, done, info)
            if nxt == 0:
                agents[0].proceed(obs, reward, done, info)
            elif nxt == 1:
                if rst == 1:
                    # First time adv plays after start of episode, set first obs
                    # to what 
                    agents[1].set_last(obs, False)
                    rst = 0
                else:
                    agents[1].proceed(obs, reward, done, info)
                    # print(agents[1].rollout_buffer.pos)
            # elif nxt == 2:
                # if ben is next, do not proceed
                # agents[1].proceed(obs, reward, done, info)
        elif curr == 1 or curr == 2:
            if done:
                # term_obs = agents[1].env.get_obs()
                agents[0].proceed(obs, reward, False, info)
                done, curr, nxt, n_steps = reset()
                rst = 1
            else:
                agents[0].proceed(obs, reward, done, info)

    # Save the trained agents
    print("Saving models...")
    interceptor.save("interceptor_model")
    adversary.save("adversary_model")
    
    # Make evaluation env
    envv = gym.make("HsjaGames-v0", steps=steps, ratio_benign=ratio, adaptive=adaptive, dataset=dataset, defended=defended, train=False, rewarder=reward, intercept = inter, seed=seed)
    # Load the trained agents
    interceptor = RPPO.load("interceptor_model", envv, "interceptor")
    adversary = RPPO.load("adversary_model", envv, "adversary")
    benign = RandomAgent(env=envv)

    mean_rint, std_rint, mean_radv, std_radv, epsilons, mean_eps, start_eps, mean_acc = evaluate_rdpolicy(interceptor, adversary, benign, envv, n_eval_episodes=100)
    
    res = np.asarray([mean_rint, std_rint, mean_radv, std_radv, mean_eps, start_eps, mean_acc])
    # np.savetxt('./logs/50benign.csv', res, delimiter=";", fmt='%1.3f')
    print(res)
    
    # df = pd.DataFrame({'mean_reward_int': mean_rint, 'std_reward_int': std_rint, 'mean_reward_adv': mean_radv,
    #                    'std_reward_adv': std_radv, 'mean_eps': mean_eps, 'start_eps': start_eps, 'mean_acc': mean_acc}, index=[0])
    # fd = pd.DataFrame(epsilons)
    # # file_path = os.path.join(logdir, 'bags_')
    # path1 = logdir + name + '.csv'
    # path2 = logdir + name + '_epsilons.csv'
    # df.to_csv(path1, index=False, float_format='%.3f')
    # fd.to_csv(path2, index=False, float_format='%.3f')
    
    return mean_eps
    
def check_full(agents):
    for i in range(2):
        # print(agents[i].rollout_buffer.pos)
        if agents[i].rollout_buffer.full:
            # print(i, "agent training")
            agents[i].close_buffer()
            agents[i].train()
            agents[i].reset_buffer()

def reset():
    return False, 1, 0, 0    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MA BAGS")
    parser.add_argument('--timesteps', default=int(4e6), type=int, help="Total number of timesteps to run for")
    parser.add_argument('--steps', default=int(5e3), type=int, help="Number of steps for each attack episode")
    parser.add_argument('--adaptive', default=int(2), type=int, help="Controls which agents are adaptive")
    parser.add_argument('--ratio', default=float(0.5), type=float, help="Probability of next draw being benign")
    parser.add_argument('--defended', default=bool(False), type=bool, help="Adversarially trained model or not")
    parser.add_argument('--seed', default=int(2), type=int, help="Seed for all PRNG sources")
    parser.add_argument('--name', default=str("bags"), type=str, help="Name for experiment")
    args = parser.parse_args()
    train(args)