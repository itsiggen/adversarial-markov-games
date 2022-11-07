import argparse
import gym
import os
import numpy as np
import optuna
import gc
import tracemalloc
from tqdm import tqdm
from agents.rppo import RPPO
from agents.benign import RandomAgent
from torchvision import datasets, transforms
from utils.evaluation import evaluate_rdpolicy
from envs.bags_games import BagsGames
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

transform=transforms.ToTensor()
dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)

def objective(trial):
    """
    TA-AD: Trained Adversary - Adaptive Defense
    """
    
    print('Training BAGS-6: TA-AD..')
    
    eval_steps = 5000
    adaptive = 3 # both adaptive 
    stt = 0 # interceptor is learning
    ratio = 0.5
    defended = False
    seed = 2

    steps = trial.suggest_categorical('steps', [1000,2000,3000])
    lr = trial.suggest_categorical('lr', [0.003,0.001,0.0003,0.0001])
    buffer = 2048
    # batch = trial.suggest_categorical('batch', [32,64,128])
    batch = 64
    # epochs = trial.suggest_categorical('epochs', [20])
    epochs = 20
    gamma = trial.suggest_float('gamma', 0.8, 0.99, step=0.01)
    ent_coef = 0
    vf_coef = 0.5
    rint = trial.suggest_categorical('rint', [3,4,5])
    scale = 8
    radv = 1
    inter = 1
    ts = trial.suggest_categorical('ts', [1e6,2e6])
    
    # Create environment
    env = gym.make("BagsGames-v0",
                   steps=steps,
                   ratio_benign=ratio,
                   adaptive=adaptive,
                   dataset=dataset,
                   scale=scale,
                   rint=rint,
                   radv=radv,
                   defended=defended,
                   intercept=inter)
    
    total_timesteps = int(ts)
    
    interceptor = RPPO(policy="MlpPolicy",
                env=env,
                agent='interceptor',
                n_steps=buffer,
                batch_size=batch,
                n_epochs=epochs,
                learning_rate=lr,
                gamma=round(gamma,2),
                tensorboard_log=None,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                verbose=0,
                seed=seed,
                # policy_kwargs=dict(net_arch=[32,32]))
                policy_kwargs=dict(net_arch=[dict(vf=[32,32], pi=[32,32])]))

    adversary = RPPO.load("mods/games/bags5adv_5.pt", env, "adversary", seed)

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
        check_full(agents, stt)
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
    interceptor.save("mods/games/bags6int_" + str(trial.number) + ".pt")
    adversary.save("mods/games/bags6adv.pt")

    # Make evaluation env
    seed = 3
    envv = gym.make("BagsGames-v0",
                    steps=eval_steps,
                    ratio_benign=ratio,
                    adaptive=adaptive,
                    dataset=dataset,
                    scale=scale,
                    defended=defended,
                    train=False,
                    rint=rint,
                    radv=radv,
                    intercept=inter)
    
    # Load the trained agents
    interceptor = RPPO.load("mods/games/bags6int_" + str(trial.number) + ".pt" , envv, "interceptor", seed)
    adversary = RPPO.load("mods/games/bags6adv.pt" , envv, "adversary", seed)
                
    benign = RandomAgent(env=envv)

    mean_rint, std_rint, mean_radv, std_radv, epsilons, lengths, mean_eps, start_eps, mean_acc = evaluate_rdpolicy(interceptor, adversary, benign, envv, n_eval_episodes=15)
    
    res = np.asarray([round(mean_rint,2), round(std_rint,2), round(mean_radv,2), round(std_radv,2), round(start_eps,3), round(mean_eps,3), round(mean_acc,3)])
    print(res)
    
    del env
    del envv
    del interceptor
    del adversary
    del benign
    gc.collect()
    
    return mean_eps, mean_acc
    
def check_full(agents, stt):
    for i in range(2):
        if agents[i].rollout_buffer.full:
        # if agents[0].rollout_buffer.full:
            # print(i, "agent training")
            agents[i].close_buffer()
            if stt == i or stt == 2:
                agents[i].train()
            agents[i].reset_buffer()

def reset():
    return False, 1, 0, 0

def test(num, scale, rew):
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
                    scale=scale,
                    defended=defended,
                    train=False,
                    rint=rew,
                    radv=1,)
    

    interceptor = RPPO.load("mods/games/bags6int_" + str(num) + ".pt", envv, "interceptor", seed)
    adversary = RPPO.load("mods/games/bags6adv.pt" , envv, "adversary", seed)

    benign = RandomAgent(env=envv)
    

    mean_rint, std_rint, mean_radv, std_radv, epsilons, iters, mean_eps, start_eps, mean_acc = evaluate_rdpolicy(interceptor, adversary, benign, envv, n_eval_episodes=100)

    res = [mean_eps, start_eps, mean_acc]

    z = list(zip(iters,epsilons))
    a = [np.interp(1000, i[0], i[1]) for i in z]
    b = [np.interp(2000, i[0], i[1]) for i in z]
    c = np.mean(a)
    d = np.mean(b)
    print(res, c, d)
        
    return mean_eps

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
    parser.add_argument('--load', default=str("3"), type=str, help="Agent to load")
    parser.add_argument('--scale', default=float(8), type=float, help="Intercept")
    parser.add_argument('--rew', default=int(4), type=bool, help="Reward used")
    args = parser.parse_args()
    if args.train:
        # Create a new optuna study.
        study = optuna.create_study(directions=['maximize', 'maximize'])
        study.optimize(objective, n_trials=1, gc_after_trial=True)
    else:
        mean_eps = test(args.load, args.scale, args.rew)
    # train(args)