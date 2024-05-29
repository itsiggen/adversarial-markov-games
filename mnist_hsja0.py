import argparse
import gym
import os
import numpy as np
import optuna
from tqdm import tqdm
from stable_baselines3 import PPO
from agents.benign import RandomAgent
from torchvision import datasets, transforms
from utils.evaluation import evaluate_policy
from envs.hsja_skip import HsjaSkip
from stable_baselines3.common.vec_env import VecNormalize
os.environ['CUDA_VISIBLE_DEVICES'] = ''

transform=transforms.ToTensor()
dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)

def objective(trial):
    
    eval_steps = 5000
    steps = trial.suggest_categorical('steps', [600,1000,3000,5000])
    buffer = 1024
    batch = 32
    lr = trial.suggest_categorical('lr', [0.003,0.001,0.0001])
    epochs = 20
    gamma = trial.suggest_float('gamma', 0.9, 0.99, step=0.01)
    ent_coef = 0
    reward = trial.suggest_categorical('reward', [1,2,3,4,5])
    ts = trial.suggest_categorical('ts', [2e4,4e4])
    defended = True
    nona = False
    seed = 2
    
    # Create environment
    env = gym.make("HsjaSkip-v0",
                   steps=steps,
                   dataset=dataset,
                   defended=defended,
                   nonadaptive=nona,
                   rewarder=reward)
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=buffer,
        batch_size=batch,
        learning_rate=round(lr,5),
        n_epochs = epochs,
        gamma=round(gamma,2),
        tensorboard_log=None,
        ent_coef = ent_coef,
        verbose=0,
        seed=seed,
        policy_kwargs=dict(net_arch=[dict(vf=[32,32], pi=[32,32])])
        )
    
    model.learn(total_timesteps=int(ts))
    model.save("mods/hsjaaskip_" + str(trial.number) + "_model.pt")

    # Evaluate the agent
    seed = 3
    envv = gym.make("HsjaSkip-v0",
                    steps=eval_steps,
                    dataset=dataset,
                    defended=defended,
                    rewarder=reward,
                    nonadaptive=nona,
                    train=False)
    
    model = PPO.load("mods/hsjaaskip_" + str(trial.number) + "_model.pt", seed=seed)
    mean_reward, std_reward, epsilons, mean_eps, start_eps, iters, mean_length, _ = evaluate_policy(model, envv, n_eval_episodes=30)
    
    res = [mean_reward, std_reward, mean_eps, start_eps, mean_length]
    print('hsjaa0:', res)
    
    return mean_eps

def test(num, rew):
    eval_steps = 5000
    defended = True
    nona = False
    seed = 2
    thres = 3    

    # Make evaluation env
    envv = gym.make("HsjaSkip-v0",
                    steps=eval_steps,
                    dataset=dataset,
                    defended=defended,
                    rewarder=rew,
                    nonadaptive=nona,
                    train=False)
        
    model = PPO.load("mods/hsjaaskip_10_model.pt", seed=seed)
    mean_reward, std_reward, epsilons, mean_eps, start_eps, iters, mean_length, _ = evaluate_policy(model, envv, n_eval_episodes=100)
    
    # attack success rate
    lsc = [i[-1] for i in epsilons]
    suc = np.sum(np.array(lsc) < thres)
    asr = suc / len(lsc)
    res = [mean_eps, start_eps, asr]
    z = list(zip(iters,epsilons))
    a = [np.interp(1000, i[0], i[1]) for i in z]
    b = [np.interp(2000, i[0], i[1]) for i in z]
    c = np.mean(a)
    d = np.mean(b)
    print(f'hsja0 | defended {defended}, nonadaptive {nona}:', res, c, d)
        
    return mean_eps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MA BAGS")
    parser.add_argument('--defended', default=bool(False), type=bool, help="Adversarially trained model or not")
    parser.add_argument('--train', default=bool(False), type=bool, help="Train or Test")
    parser.add_argument('--load', default=str("10"), type=str, help="Model to load")
    parser.add_argument('--rew', default=int(5), type=bool, help="Reward used")
    args = parser.parse_args()
    if args.train:
        # Create a new optuna study.
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, gc_after_trial=True)
    else:
        mean_eps = test(args.load, args.rew)