import argparse
import gym
import os
import numpy as np
import optuna
from tqdm import tqdm
from agents.rppo import RPPO
from agents.benign import RandomAgent
from envs.bags_games_cifar import BagsGamesCIFAR
from torchvision import datasets, transforms
from utils.evaluation import evaluate_rdpolicy

transform = transforms.ToTensor()
dataset = datasets.CIFAR10('data', train=False, transform=transform, download=True)

def objective(trial):
    """
    VA-VD: Adaptive Adversary - No Defense
    """
    
    print('Training CBAGS-3: AA-VD..')   
    
    adaptive = 1 # adv adaptive, vanilla defense 
    vanilla = True
    ratio = 0.5
    stt = 1 # adversary is learning
    inter = 1
    defended = False
    seed = 2
    
    eval_steps = 5000
    steps = trial.suggest_categorical('steps', [1000,2000,3000])
    buffer = 2048
    batch = trial.suggest_categorical('batch', [32,64,128])
    lr = trial.suggest_categorical('lr', [0.003,0.001,0.0003,0.0001])
    epochs = 20
    gamma = trial.suggest_float('gamma', 0.85, 0.99, step=0.01)
    ent_coef = trial.suggest_categorical('ent_coef', [0,0.00001])
    scale = trial.suggest_categorical('scale', [5,10,20])
    radv = trial.suggest_categorical('reward', [1,2,3,4,5])
    ts = trial.suggest_categorical('ts', [1e6,2e6])
    # ts = 6e5


    # Create environment
    env = gym.make("BagsGamesCIFAR-v0",
                   steps=steps,
                   ratio_benign=ratio,
                   adaptive=adaptive,
                   vanilla=vanilla,
                   dataset=dataset,
                   train=False,
                   scale=scale,
                   rint=5,
                   radv=radv,
                   defended=defended,
                   intercept=inter)
    
    total_timesteps = int(ts)

    interceptor = RPPO.load("mods/games/bags6int_0.pt", env, "interceptor", seed)

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
    
    # env.contrasts.save()
    
    print("Saving models...")

    adversary.save("mods/games/cbags3adv_" + str(trial.number) + ".pt")

    seed = 3
    envv = gym.make("BagsGamesCIFAR-v0",
                    steps=eval_steps,
                    ratio_benign=ratio,
                    adaptive=adaptive,
                    dataset=dataset,
                    defended=defended,
                    train=False,
                    scale=scale,
                    rint=1,
                    radv=radv)
    
    # Load the trained agents
    interceptor = RPPO.load("mods/games/bags6int_0.pt", envv, "interceptor", seed)
    adversary = RPPO.load("mods/games/bags3adv_" + str(trial.number) + ".pt" , envv, "adversary", seed)
                
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
                agents[i].train()
            agents[i].reset_buffer()

def reset():
    return False, 1, 0, 0


def test(num, rew, scale):
    eval_steps = 5000
    adaptive = 1
    vanilla = True
    defended = True
    ratio = 0.5
    seed = 2
    
    # Make evaluation env
    envv = gym.make("BagsGamesCIFAR-v0",
                    steps=eval_steps,
                    ratio_benign=ratio,
                    adaptive=adaptive,
                    vanilla=vanilla,
                    dataset=dataset,
                    defended=defended,
                    train=False,
                    scale=scale,
                    rint=1,
                    radv=rew)
        
    
    interceptor = RPPO.load("mods/games/bags6int_0.pt", envv, "interceptor", seed)
    # adversary = RPPO.load("mods/games/bags3adv_" + str(num) + ".pt" , envv, "adversary", seed)
    adversary = RPPO.load("mods/cifarbags_" + str(num) + "_model.pt", envv, "adversary", seed=seed)

    benign = RandomAgent(env=envv)
    
    
    mean_rint, std_rint, mean_radv, std_radv, epsilons, iters, mean_eps, start_eps, mean_acc = evaluate_rdpolicy(interceptor, adversary, benign, envv, n_eval_episodes=100)

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
    parser.add_argument('--train', default=bool(True), type=bool, help="Train or Test")
    parser.add_argument('--load', default=str("16"), type=str, help="Model to load")
    parser.add_argument('--rew', default=int(3), type=bool, help="Reward used")
    parser.add_argument('--scale', default=int(10), type=bool, help="Scale used")
    args = parser.parse_args()
    if args.train:
        # Create a new optuna study.
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20, gc_after_trial=True)
    else:
        mean_eps, mean_acc = test(args.load, args.rew, args.scale)
    # train(args)