import argparse
import gym
import os
import gc
import pandas as pd
import numpy as np
import optuna
from tqdm import tqdm
from agents.rppo import RPPO
from agents.benign import RandomAgent
from torchvision import datasets, transforms
from utils.evaluation import evaluate_rpolicy
from envs.bags_games import BagsGames
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

transform=transforms.ToTensor()
dataset = datasets.MNIST('data', train=False, transform=transform, download=True)

def objective(trial):
    """
    VA-AD: Vanilla Adversary - Adaptive Defense
    """
    
    print('Training BAGS-5: VA-AD..')
    
    eval_steps = 5000
    adaptive = 3 # int adaptive
    ratio = 0.5
    defended = False
    seed = 2
    logdir = "./logs/"
    
    steps = trial.suggest_categorical('steps', [1000,2000,3000])
    lr = trial.suggest_categorical('lr', [0.003,0.001,0.0003,0.0001])
    buffer = 2048
    batch = trial.suggest_categorical('batch', [32,64,128])
    epochs = 20
    gamma = trial.suggest_float('gamma', 0.85, 0.99, step=0.01)
    ent_coef = 0
    vf_coef = 0.5
    scale = trial.suggest_categorical('scale', [3,5,8,10,20])
    inter = 1
    radv = trial.suggest_categorical('radv', [2,3,4])
    # radv = 3
    rint = 3
    ts = trial.suggest_categorical('ts', [1e6,2e6])
    # ts = 6e5

    # Create environment
    env = gym.make("BagsGames-v0",
                   steps=steps,
                   ratio_benign=ratio,
                   adaptive=adaptive,
                   dataset=dataset,
                   train=False,
                   scale=scale,
                   rint=rint,
                   radv=radv,
                   defended=defended,
                   intercept=inter)
    
    total_timesteps = int(ts)
    
    interceptor = RPPO.load("mods/games/bags4nint_5.pt" , env, "interceptor", seed)
    
    adversary = RPPO(policy="MlpPolicy",
                env=env,
                agent='adversary',
                n_steps=buffer,
                batch_size=batch,
                n_epochs=epochs,
                learning_rate=lr,
                gamma=round(gamma,2),
                tensorboard_log=None,
                ent_coef = ent_coef,
                vf_coef=vf_coef,
                verbose=0,
                seed=seed,
                policy_kwargs=dict(net_arch=[dict(vf=[32,32], pi=[32,32])]))
    
    benign = RandomAgent(env=env)
      
    agents = [interceptor, adversary, benign]
   
    
    for agent in agents:
        agent.setup_learn()
    obs = env.reset()
    agents[0].set_last(obs, False)
    done = False
    curr, nxt = 1, 0
    n_steps = 0
        
    for timestep in tqdm(range(total_timesteps), disable=False):
        # Check if a rollout buffer has been filled and train
        check_full(agents)
        # Store previous move
        prev = curr
        # next agent moves
        # print(nxt)
        # obs, reward, done, info, curr, nxt = agents[nxt].move()
        obs, reward, done, info = agents[nxt].move()
        curr = info["curr"]
        nxt = info["next"]
        n_steps += 1
        
        if curr == 0:
            if n_steps == 1:
                # env has been just reset
                agents[1].set_last(obs, False)
            else:
                agents[prev].proceed(obs, reward, done, info)
        elif curr == 1 or curr == 2:
            if done:
                # term_obs = agents[1].env.get_obs()
                agents[0].proceed(obs, reward, False, info)
                # print(info['gap'], info['epsilon'], info['correct'])
                # agents[0].set_last(obs, False)
                done, curr, nxt, n_steps = reset()
            else:
                agents[0].proceed(obs, reward, done, info)

    # Save the trained agents
    print("Saving models...")
    adversary.save("mods/games/bags5adv_" + str(trial.number) + ".pt")
    
    # Make evaluation env
    seed = 3
    envv = gym.make("BagsGames-v0",
                    steps=eval_steps,
                    ratio_benign=ratio,
                    adaptive=adaptive,
                    dataset=dataset,
                    defended=defended,
                    train=False,
                    scale=scale,
                    rint=rint,
                    radv=radv,
                    intercept=inter)

    
    interceptor = RPPO.load("mods/games/bags4nint_5.pt" , envv, "interceptor", seed)
    adversary = RPPO.load("mods/games/bags5adv_" + str(trial.number) + ".pt" , envv, "adversary", seed)
                
    benign = RandomAgent(env=envv)

    mean_rint, std_rint, mean_radv, std_radv, epsilons, lengths, mean_eps, start_eps, mean_acc = evaluate_rpolicy(interceptor, adversary, benign, envv, n_eval_episodes=15)
    # envv.gstates.save("mods/data/hsja4eval_" + str(trial.number) + ".csv")

    res = np.asarray([round(mean_rint,2), round(std_rint,2), round(mean_radv,2), round(std_radv,2), round(start_eps,3), round(mean_eps,3), round(mean_acc,3)])
    # np.savetxt('./logs/50benign.csv', res, delimiter=";", fmt='%1.3f')
    print(res)
    
    del env
    del envv
    del interceptor
    del adversary
    del benign
    
    return mean_eps
    
def check_full(agents):
    for i in range(2):
        if agents[i].rollout_buffer.full:
        # if agents[0].rollout_buffer.full:
            # print(i, "agent training")
            agents[i].close_buffer()
            agents[i].train()
            agents[i].reset_buffer()

def reset():
    return False, 1, 0, 0

def test(num, inter, rew, scale):
    eval_steps = 5000
    adaptive = 3 # int adaptive 
    ratio = 0.5
    defended = False
    seed = 2

    # Make evaluation env
    envv = gym.make("BagsGames-v0",
                    steps=eval_steps,
                    ratio_benign=ratio,
                    adaptive=adaptive,
                    dataset=dataset,
                    defended=defended,
                    scale=scale,
                    train=False,
                    rint=rew,
                    radv=1,
                    intercept=inter)
    
    # Load the trained agents
    interceptor = RPPO.load("mods/games/bags4nint_5.pt" , envv, "interceptor", seed)
    adversary = RPPO.load("mods/games/bags5adv_" + str(num) + ".pt", envv, "adversary", seed)

    benign = RandomAgent(env=envv)


    mean_rint, std_rint, mean_radv, std_radv, epsilons, iters, mean_eps, start_eps, mean_acc = evaluate_rpolicy(interceptor, adversary, benign, envv, n_eval_episodes=100)
    
    res = [mean_eps, start_eps, mean_acc]

    c = np.mean([i[1000] for i in epsilons])
    d = np.mean([i[2000] for i in epsilons])
    print('bags3:', res, c, d)
        
    return mean_eps, mean_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MA BAGS")
    parser.add_argument('--timesteps', default=int(4e6), type=int, help="Total number of timesteps to run for")
    parser.add_argument('--steps', default=int(5e3), type=int, help="Number of steps for each attack episode")
    parser.add_argument('--adaptive', default=int(2), type=int, help="Controls which agents are adaptive")
    parser.add_argument('--ratio', default=float(0.5), type=float, help="Probability of next draw being benign")
    parser.add_argument('--defended', default=bool(False), type=bool, help="Adversarially trained model or not")
    parser.add_argument('--seed', default=int(2), type=int, help="Seed for all PRNG sources")
    parser.add_argument('--name', default=str("false"), type=str, help="Name for experiment")
    parser.add_argument('--train', default=bool(False), type=bool, help="Train or Test")
    parser.add_argument('--load', default=str("2"), type=str, help="Model to load")
    parser.add_argument('--inter', default=float(1), type=float, help="Intercept")
    parser.add_argument('--scale', default=float(10), type=float, help="Scale")
    parser.add_argument('--rew', default=int(4), type=bool, help="Reward used")
    args = parser.parse_args()
    if args.train:
        # Create a new optuna study.
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, n_jobs=1, gc_after_trial=True)
    else:
        mean_eps, mean_acc = test(args.load, args.inter, args.rew, args.scale)
    # train(args)