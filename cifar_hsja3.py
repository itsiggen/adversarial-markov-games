import argparse
import gym
import os
import pandas as pd
import numpy as np
import optuna
from tqdm import tqdm
from agents.rppo import RPPO
from agents.benign import RandomAgent
from torchvision import datasets, transforms
from utils.evaluation import evaluate_rdpolicy
from envs.hsja_games_cifar import HsjaGamesCIFAR
from stable_baselines3.common.vec_env import VecNormalize
os.environ['CUDA_VISIBLE_DEVICES'] = ''

transform = transforms.ToTensor()
dataset = datasets.CIFAR10('data', train=False, transform=transform, download=True)

def objective(trial):
    
    adaptive = 1 # adv adaptive, plus stateful defense
    ratio = 0.5
    vanilla = True
    stt = 1 # adversary is learning
    cont = 0
    inter = 1
    defended = False
    seed = 2
    
    eval_steps = 5000
    steps = trial.suggest_categorical('steps', [600,1000,3000])
    # steps = 300
    buffer = trial.suggest_categorical('buffer', [64,128,256,1024])
    batch = trial.suggest_categorical('batch', [16,32])
    # batch = 32
    lr = trial.suggest_categorical('lr', [0.003,0.001,0.0003,0.0001])
    epochs = 20
    gamma = trial.suggest_float('gamma', 0.8, 0.99, step=0.01)
    ent_coef = 0
    scale = trial.suggest_categorical('scale', [1,2,4,8,16])
    # scale = 8
    radv = trial.suggest_categorical('reward', [2,3,4,5,6,7,8])
    ts = trial.suggest_categorical('ts', [1e6,2e6])
    # ts = 100


    # Create environment
    env = gym.make("HsjaGamesCIFAR-v0",
                   steps=steps,
                   ratio_benign=ratio,
                   adaptive=adaptive,
                   dataset=dataset,
                   scale=scale,
                   train=False,
                   vanilla=vanilla,
                   cont=cont,
                   rint=5,
                   radv=radv,
                   defended=defended,
                   intercept=inter)
    
    total_timesteps = int(ts)

    interceptor = RPPO.load("mods/games/hsja4int_8.pt" , env, "interceptor", seed)
    # interceptor.mode = 0

    # print(env.action_space)
    adversary = RPPO(policy="MlpPolicy",
                env=env,
                agent='adversary',
                n_steps=buffer,
                batch_size=batch,
                n_epochs=epochs,
                learning_rate=lr, # 0.00039
                gamma=round(gamma,2),
                tensorboard_log=None,
                ent_coef=ent_coef, # 0.0001
                verbose=0,
                seed=seed,
                policy_kwargs=dict(net_arch=[32,32]))
                # policy_kwargs=dict(net_arch=[dict(vf=[32,32], pi=[32,32])]))
    
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
        
    for timestep in tqdm(range(total_timesteps), disable=True):
        # Check if a rollout buffer has been filled and train
        check_full(agents, stt)
        # Store previous move
        prev = curr
        # next agent moves
        # print(nxt)
        obs, reward, done, info = agents[nxt].move()
        curr = info["curr"]
        nxt = info["next"]
        # print(curr,nxt)
        n_steps += 1
        # print("cadence", prev, curr, nxt, reward)

        if curr == 0:
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
    
    # env.contrasts.save()
    
    print("Saving models...")

    adversary.save("mods/chsja3adv_" + str(trial.number) + ".pt")

    seed = 3
    envv = gym.make("HsjaGamesCIFAR-v0",
                    steps=eval_steps,
                    ratio_benign=ratio,
                    adaptive=adaptive,
                    dataset=dataset,
                    scale=scale,
                    defended=defended,
                    train=False,
                    test=True,
                    vanilla=vanilla,
                    cont=cont,
                    rint=5,
                    radv=radv,
                    intercept=inter)
    
    # Load the trained agents
    interceptor = RPPO.load("mods/games/hsja4int_8.pt", envv, "interceptor", seed)
    adversary = RPPO.load("mods/chsja3adv_" + str(trial.number) + ".pt" , envv, "adversary", seed)
                
    benign = RandomAgent(env=envv)
        
    mean_rint, std_rint, mean_radv, std_radv, epsilons, lengths, mean_eps, start_eps, mean_acc = evaluate_rdpolicy(interceptor, adversary, benign, envv, n_eval_episodes=15)
    # envv.gstates.save("mods/data/hsja4eval_" + str(trial.number) + ".csv")
    
    res = [mean_radv, std_radv, mean_eps, start_eps, mean_acc]
    print(res)
    
    return mean_eps

def check_full(agents, stt):
    for i in range(2):
        if agents[i].rollout_buffer.full:
        # if agents[0].rollout_buffer.full:
            # print(i, "agent training")
            agents[i].close_buffer()
            if stt == i or stt == 2:
                # print(agents[i].agent)
                agents[i].train()
            agents[i].reset_buffer()

def reset():
    return False, 1, 0, 0


def test(num, rew, scale):
    eval_steps = 5000
    adaptive = 1
    vanilla = True
    defended = False
    ratio = 0.5
    cont = 0
    seed = 2
    inter = 1
    
    # Make evaluation env
    envv = gym.make("HsjaGamesCIFAR-v0",
                    steps=eval_steps,
                    ratio_benign=ratio,
                    adaptive=adaptive,
                    vanilla=vanilla,
                    dataset=dataset,
                    scale=scale,
                    defended=defended,
                    cont=cont,
                    train=False,
                    test=True,
                    rint=rew,
                    radv=1,
                    intercept=inter)
        
    
    interceptor = RPPO.load("mods/games/hsja4int_8.pt", envv, "interceptor", seed)
    # adversary = RPPO.load("mods/chsja3adv_" + str(num) + ".pt" , envv, "adversary", seed)
    adversary = RPPO.load("mods/chsja3adv_" + str(num) + ".pt" , envv, "adversary", seed)
    
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
    parser.add_argument('--train', default=bool(True), type=bool, help="Train or Test")
    parser.add_argument('--load', default=str("11"), type=str, help="Model to load")
    parser.add_argument('--scale', default=int(10), type=int, help="Scale")
    parser.add_argument('--rew', default=int(5), type=bool, help="Reward used")
    args = parser.parse_args()
    if args.train:
        # Create a new optuna study.
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30, gc_after_trial=True)
    else:
        mean_eps = test(args.load, args.rew, args.scale)
    # train(args)