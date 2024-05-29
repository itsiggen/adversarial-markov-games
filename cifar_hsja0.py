import argparse
import gym
import pandas as pd
import numpy as np
import os, sys
import optuna
from envs.hsja_skip_cifar import HsjaSkipCIFAR
from stable_baselines3 import PPO
from torchvision import datasets, transforms
from utils.evaluation import evaluate_policy


transform = transforms.ToTensor()
dataset = datasets.CIFAR10('./data', train=False, transform=transform, download=True)

def objective(trial):
    """
    AA-ND: Adaptive Adversary - No Defense
    """
    
    print('Training CHSJA-1: AA-ND..')  
    
    eval_steps = 5000
    steps = trial.suggest_categorical('steps', [1000,3000,5000])
    arch = trial.suggest_categorical('architecture', [32,64,128])
    buffer = trial.suggest_categorical('buffer', [256,1024,2048])
    batch = trial.suggest_categorical('batch', [32,64,128])
    lr = trial.suggest_categorical('lr', [0.001,0.0005,0.0001,0.00005])
    gamma = trial.suggest_float('gamma', 0.85, 0.99, step=0.01)
    ent_coef = trial.suggest_categorical('ent_coef', [0,0.001,0.0001])
    reward = trial.suggest_categorical('reward', [1,2,3,4,5])
    ts = trial.suggest_categorical('timesteps', [1e4,2e4,5e4])
    
    defended = False
    nona = False
    seed = 2
    
    # Create environment
    env = gym.make("HsjaSkipCIFAR-v0",
                   steps=steps,
                   dataset=dataset,
                   defended=defended,
                   nonadaptive=nona,
                   rewarder=reward)
    """
    Best parameters: {'architecture': 32, 'buffer': 1024, 'batch': 64, 'sde': False,
                      'lr': 0.00039, 'gamma': 0.92, 'ent_coef': 0.0001, 'reward': 4}
    Best median_epsilon: 2.76
    """
        
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=buffer,
        batch_size=batch,
        learning_rate=round(lr,5),
        gamma=round(gamma,2),
        tensorboard_log=None,
        ent_coef = round(ent_coef,5),
        # use_sde=sde,
        verbose=0,
        seed=2,
        policy_kwargs=dict(net_arch=[dict(vf=[arch,arch], pi=[arch,arch])])
    )
    
    model.learn(total_timesteps=int(ts))
    model.save("mods/cifarhsja_" + str(trial.number) + "_model.pt")

    # Evaluate the agent
    seed = 3
    envv = gym.make('HsjaSkipCIFAR-v0',
                    steps=eval_steps,
                    dataset=dataset,
                    defended=defended,
                    nonadaptive=nona,
                    rewarder=reward,
                    train=False)
    
    model = PPO.load("mods/cifarhsja_" + str(trial.number) + "_model.pt", seed=seed)
    mean_reward, std_reward, epsilons, mean_eps, start_eps, iters, mean_length, _ = evaluate_policy(model, envv, n_eval_episodes=100)
    
    res = np.asarray([mean_reward, std_reward, mean_eps, start_eps, mean_length])
    print('chsja1:', res)
    
    return mean_eps

def test(num, rew):
    eval_steps = 5000
    defended = True
    nona = False
    seed = 2
    thres = 3

    # Make evaluation env
    envv = gym.make('HsjaSkipCIFAR-v0',
                    steps=eval_steps,
                    dataset=dataset,
                    defended=defended,
                    rewarder=rew,
                    nonadaptive=nona,
                    train=False,
                    test=True)
        
    model = PPO.load("mods/cifarhsja_" + str(num) + "_model.pt", seed=seed)
    mean_reward, std_reward, epsilons, mean_eps, start_eps, iters, mean_length, _ = evaluate_policy(model, envv, n_eval_episodes=100)
    
    # attack success rate
    lsc = [i[-1] for i in epsilons]
    suc = np.sum(np.array(lsc) < thres)
    asr = suc / len(lsc)
    res = [mean_reward, std_reward, mean_eps, start_eps, mean_length, asr]
    z = list(zip(iters,epsilons))
    a = [np.interp(1000, i[0], i[1]) for i in z]
    b = [np.interp(2000, i[0], i[1]) for i in z]
    c = np.mean(a)
    d = np.mean(b)
    print(f'chsja0 | defended {defended}, not-adaptive {nona}:', res, c, d)
        
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
    parser.add_argument('--load', default=str("5"), type=str, help="Model to load")
    parser.add_argument('--rew', default=int(4), type=bool, help="Reward used")
    args = parser.parse_args()
    if args.train:
        # Create a new optuna study.
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, gc_after_trial=True)
    else:
        mean_eps = test(args.load, args.rew)
    # train(args)