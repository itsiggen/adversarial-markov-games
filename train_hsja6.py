import argparse
import gym
import os
import pandas as pd
import numpy as np
import optuna
import gc
import tracemalloc
from tqdm import tqdm
from agents.rppo import RPPO
from agents.benign import RandomAgent
from torchvision import datasets, transforms
from utils.evaluation import evaluate_rdpolicy
from envs.hsja_games import HsjaGames
from stable_baselines3.common.vec_env import VecNormalize
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

transform=transforms.ToTensor()
dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)

# tracemalloc.start()

def objective(trial):
    """
    TA-AD: Trained Adversary - Adaptive Defense
    """
    
    print('Training HSJA-6: TA-AD..')
    
    eval_steps = 5000
    adaptive = 3 # both adaptive 
    ratio = 0.5
    defended = False
    cont = 2 # contrastive model used
    seed = 2

    steps = 5000
    lr = trial.suggest_categorical('lr', [0.001,0.003,0.0001])
    # buffer = trial.suggest_categorical('buffer', [256,512,1024])
    buffer = 2048
    # batch = trial.suggest_categorical('batch', [64,512,2048])
    batch = 64
    # epochs = trial.suggest_categorical('epochs', [20])
    epochs = 20
    gamma = trial.suggest_float('gamma', 0.9, 0.99, step=0.01)
    # ent_coef = trial.suggest_categorical('ent_coef', [0,0.001,0.0001])
    ent_coef = 0
    vf_coef = 0.5
    que = True
    rint = trial.suggest_categorical('rint', [1,2,3,4,5,6,7,8])
    radv = 1
    inter = 1
    ts = 1e6
    norm = False
    
    # Create environment
    env = gym.make("HsjaGames-v0",
                   steps=steps,
                   ratio_benign=ratio,
                   adaptive=adaptive,
                   dataset=dataset,
                   train=que,
                   rint=rint,
                   radv=radv,
                   defended=defended,
                   cont=cont,
                   intercept=inter)
    
    total_timesteps = int(ts)
    
    interceptor = RPPO(policy="MlpPolicy",
                env=env,
                agent='interceptor',
                n_steps=buffer,
                batch_size=batch,
                n_epochs=epochs,
                learning_rate=lr, # 0.00039
                gamma=round(gamma,2), # 0.92
                tensorboard_log=None,
                ent_coef=ent_coef, # 0.0001
                vf_coef=vf_coef,
                verbose=0,
                seed=seed,
                normalize=norm,
                # policy_kwargs=dict(net_arch=[32,32]))
                policy_kwargs=dict(net_arch=[dict(vf=[32,32], pi=[32,32])]))

    adversary = RPPO.load("mods/games/hsja5adv_9.pt", env, "adversary", seed)

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
        

    # snapshot1 = tracemalloc.take_snapshot()
    # ... call the function leaking memory ...
    # snapshot2 = tracemalloc.take_snapshot()
    
    for timestep in tqdm(range(total_timesteps), disable=False):
        # Check if a rollout buffer has been filled and train
        check_full(agents)
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
                
                # snapshot2 = tracemalloc.take_snapshot()
                # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
                # print("[ Top 10 differences ]")
                # for stat in top_stats[:10]:
                #     print(stat)
                # # snapshot1 = tracemalloc.take_snapshot()
                                
                # term_obs = agents[1].env.get_obs()
                agents[0].proceed(obs, reward, False, info)
                done, curr, nxt, n_steps = reset()
                rst = 1
            else:
                agents[0].proceed(obs, reward, done, info)

    # Save the trained agents
    
    # env.contrasts.save()
    
    print("Saving models...")
    interceptor.save("mods/games/hsja6int_" + str(trial.number) + ".pt")
    adversary.save("mods/games/hsja6adv.pt")
    # env.save("mods/games/normed_" + str(trial.number) + ".pkl")
    # env.gstates.save("mods/data/hsja4_" + str(trial.number) + ".csv")
    # if norm:
    #     interceptor.RecNorm.save("mods/games/inormed_" + str(trial.number) + ".pkl")
    #     adversary.RecNorm.save("mods/games/anormed_" + str(trial.number) + ".pkl")
    
    
    # Make evaluation env
    # envv = gym.make("HsjaGames-v0", steps=eval_steps, ratio_benign=ratio, adaptive=adaptive, dataset=dataset, defended=defended, train=False, rint=rint, radv=radv, intercept=inter)
    # Different seed for validation pairs.
    seed = 3
    envv = gym.make("HsjaGames-v0",
                    steps=eval_steps,
                    ratio_benign=ratio,
                    adaptive=adaptive,
                    dataset=dataset,
                    defended=defended,
                    cont=cont,
                    train=False,
                    rint=rint,
                    radv=radv,
                    intercept=inter)
    # Load the trained agents

    interceptor = RPPO.load("mods/games/hsja6int_" + str(trial.number) + ".pt" , envv, "interceptor", seed)
    adversary = RPPO.load("mods/games/hsja6adv.pt" , envv, "interceptor", seed)
                
    benign = RandomAgent(env=envv)

    mean_rint, std_rint, mean_radv, std_radv, epsilons, lengths, mean_eps, start_eps, mean_acc = evaluate_rdpolicy(interceptor, adversary, benign, envv, n_eval_episodes=30)
    # envv.gstates.save("mods/data/hsja4eval_" + str(trial.number) + ".csv")
    
    res = np.asarray([round(mean_rint,2), round(std_rint,2), round(mean_radv,2), round(std_radv,2), round(start_eps,3), round(mean_eps,3), round(mean_acc,3)])
    print(res)
    
    del env
    del envv
    del interceptor
    del adversary
    del benign
    gc.collect()
    
    return mean_eps, mean_acc
    
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

def test(num, inter, rew):
    eval_steps = 5000
    adaptive = 3 # int adaptive 
    ratio = 0.5
    defended = False
    cont = 2
    seed = 2

    # Make evaluation env
    envv = gym.make("HsjaGames-v0",
                    steps=eval_steps,
                    ratio_benign=ratio,
                    adaptive=adaptive,
                    dataset=dataset,
                    defended=defended,
                    cont=cont,
                    train=False,
                    rint=rew,
                    radv=1,
                    intercept=inter)
    

    interceptor = RPPO.load("mods/games/hsja6int_" + str(num) + ".pt", envv, "interceptor", seed)
    adversary = RPPO.load("mods/games/hsja6adv.pt" , envv, "adversary", seed)

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
    parser.add_argument('--load', default=str("32"), type=str, help="Agent to load")
    parser.add_argument('--inter', default=float(1), type=float, help="Intercept")
    parser.add_argument('--rew', default=int(5), type=bool, help="Reward used")
    args = parser.parse_args()
    if args.train:
        # Create a new optuna study.
        study = optuna.create_study(directions=['maximize', 'maximize'])
        study.optimize(objective, n_trials=1, gc_after_trial=True)
    else:
        mean_eps, mean_acc = test(args.load, args.inter, args.rew)
    # train(args)