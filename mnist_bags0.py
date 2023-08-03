import argparse
import gym
import os
import numpy as np
import optuna
from tqdm import tqdm
from stable_baselines3 import PPO
from envs.bags_skip import BagsSkip
from torchvision import datasets, transforms
from utils.evaluation import evaluate_policy
os.environ['CUDA_VISIBLE_DEVICES'] = ''

transform=transforms.ToTensor()
dataset = datasets.MNIST('data', train=False, transform=transform, download=True)

def objective(trial):
    """
    AA-ND: Adaptive Adversary - No Defense
    """
    
    print('Training BAGS-1: AA-ND..')    
    
    eval_steps = 5000
    steps = trial.suggest_categorical('steps', [600,1000,3000,5000])
    buffer = 2048
    batch = 128
    lr = trial.suggest_categorical('lr', [0.001,0.0005,0.0001])
    ent_coef = trial.suggest_categorical('ent_coef', [0,0.0001,0.00005])
    epochs = 20
    gamma = trial.suggest_float('gamma', 0.8, 0.99, step=0.01)
    ent_coef = 0
    scale = trial.suggest_categorical('scale', [5,10,20,25])
    reward = trial.suggest_categorical('reward', [1,2,3,4,5])
    ts = trial.suggest_categorical('ts', [1e5,1e6,5e6])
    defended = False
    nona = False
    seed = 2
    
    """
    Best: 2.468632221221924 with parameters:
    {'architecture': 32, 'buffer': 2048, 'batch': 128, 'steps': 1000, 'total': 5000000.0,
    'lr': 0.0001, 'epochs': 20, 'gamma': 0.82, 'ent_coef': 5e-05, 'scale': 25, 'reward': 3}
    """
    
    # Create environment
    env = gym.make("BagsSkip-v0",
                   steps=steps,
                   dataset=dataset,
                   defended=defended,
                   nonadaptive=nona,
                   scale=scale,
                   rewarder=reward)
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=buffer,
        batch_size=batch,
        learning_rate=round(lr,5),
        n_epochs=epochs,
        gamma=round(gamma,2),
        tensorboard_log=None,
        ent_coef=round(ent_coef,5),
        verbose=0,
        policy_kwargs=dict(net_arch=[dict(vf=[32,32], pi=[32,32])])
        )
    model.learn(total_timesteps=int(ts))
    model.save("mods/bagsskip_" + str(trial.number) + "_model.pt")

    # Evaluate the agent
    seed = 3
    envv = gym.make("BagsSkip-v0",
                    steps=eval_steps,
                    dataset=dataset,
                    defended=defended,
                    nonadaptive=nona,
                    train=False,
                    scale=scale,
                    rewarder=reward)
    
    model = PPO.load("mods/bagsskip_" + str(trial.number) + "_model.pt", seed)
    mean_reward, std_reward, epsilons, mean_eps, start_eps, iters, mean_length, _ = evaluate_policy(model, envv, n_eval_episodes=30)
    
    res = [mean_reward, std_reward, mean_eps, start_eps]
    print('bags0:', res)
    
    return mean_eps

def test(num, rew):
    eval_steps = 5000
    defended = False
    nona = True
    seed = 2

    # Make evaluation env
    envv = gym.make("BagsSkip-v0",
                    steps=eval_steps,
                    dataset=dataset,
                    defended=defended,
                    rewarder=rew,
                    nonadaptive=nona,
                    train=False)
        
    model = PPO.load("mods/bagsskip_" + str(num) + "_model.pt", seed=seed)
    mean_reward, std_reward, epsilons, mean_eps, start_eps, iters, mean_length, _ = evaluate_policy(model, envv, n_eval_episodes=100)
    
    res = [mean_reward, std_reward, mean_eps, start_eps, mean_length]
    # z = list(zip(iters,epsilons))
    # a = [np.interp(1000, i[0], i[1]) for i in z]
    # b = [np.interp(2000, i[0], i[1]) for i in z]
    # print(epsilons[0][4999])
    # print(len(epsilons[0]))
    # print(len(epsilons[1]))
    c = np.mean([i[1000] for i in epsilons])
    d = np.mean([i[2000] for i in epsilons])
    # c = np.mean(a)
    # d = np.mean(b)
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
    parser.add_argument('--load', default=str("49"), type=str, help="Model to load")
    parser.add_argument('--rew', default=int(3), type=bool, help="Reward used")
    args = parser.parse_args()
    if args.train:
        # Create a new optuna study.
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, gc_after_trial=True)
    else:
        mean_eps = test(args.load, args.rew)
    # train(args)
