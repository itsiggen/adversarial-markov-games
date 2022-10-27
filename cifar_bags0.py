import argparse
import gym
import numpy as np
import os, sys
import optuna
from envs.bags_skip_cifar import BagsSkipCIFAR
from stable_baselines3 import PPO
from torchvision import datasets, transforms
from utils.evaluation import evaluate_policy

transform = transforms.ToTensor()
dataset = datasets.CIFAR10('./data', train=False, transform=transform, download=True)

def objective(trial):
    """
    AA-ND: Adaptive Adversary - No Defense
    """
    
    print('Training CBAGS-1: AA-ND..')   
    
    eval_steps = 5000
    arch = trial.suggest_categorical('architecture', [32,64,128])
    buffer = 2048
    batch = trial.suggest_categorical('batch', [32,64,128])
    steps = trial.suggest_categorical('steps', [1000,2000,3000])
    lr = trial.suggest_categorical('lr', [0.001,0.0005,0.0001,0.00005])
    epochs = 20
    gamma = trial.suggest_float('gamma', 0.85, 0.99, step=0.01)
    ent_coef = trial.suggest_categorical('ent_coef', [0,0.001,0.0005,0.0001,0.00005])
    scale = trial.suggest_categorical('scale', [5,10,20,25])
    reward = trial.suggest_categorical('reward', [1,2,3,4,5]) #3 is best
    ts = trial.suggest_categorical('total', [1e5,1e6,5e6])
    
    defended = False
    nona = False
    seed = 2
    
    # Create environment
    env = gym.make("BagsSkipCIFAR-v0",
                   steps=steps,
                   dataset=dataset,
                   defended=defended,
                   nonadaptive=nona,
                   rewarder=reward,
                   scale=scale)
    """
    Best parameters: {'architecture': 32, 'buffer': 256, 'batch': 64, 'lr': 5e-05,
                      'gamma': 0.9, 'ent_coef': 0, 'scale': 5, 'reward': 3}
    2nd best:       {'architecture': 32, 'buffer': 256, 'batch': 32, 'lr': 5e-05,
                      'gamma': 0.83, 'ent_coef': 0, 'scale': 5, 'reward': 1}
    Best mean_epsilon: 1.458
    2nd best:          1.552
    """
       
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=buffer,
        batch_size=batch,
        learning_rate=round(lr,5),
        n_epochs = epochs,
        gamma=round(gamma,2),
        tensorboard_log=None,
        ent_coef=round(ent_coef,5),
        verbose=0,
        seed=seed,
        policy_kwargs=dict(net_arch=[dict(vf=[arch,arch], pi=[arch,arch])])
    )
    
    model.learn(total_timesteps=int(ts))
    model.save("mods/cifarbags_" + str(trial.number) + "_model.pt")

    # Evaluate the agent
    seed = 3
    envv = gym.make("BagsSkipCIFAR-v0",
                    steps=eval_steps,
                    dataset=dataset,
                    defended=defended,
                    nonadaptive=nona,
                    rewarder=reward,
                    scale=scale,
                    train=False)

    model = PPO.load("mods/cifarbags_" + str(trial.number) + "_model.pt", seed=seed)
    mean_reward, std_reward, epsilons, mean_eps, start_eps, iters, mean_length, _ = evaluate_policy(model, envv, n_eval_episodes=15)
       
    res = [mean_reward, std_reward, mean_eps, start_eps]
    print('cbags1:', res)
    
    return mean_eps

def test(num, rew, scale):
    eval_steps = 5000
    defended = False
    nona = False
    seed = 2

    # Make evaluation env
    envv = gym.make("BagsSkipCIFAR-v0",
                    steps=eval_steps,
                    dataset=dataset,
                    defended=defended,
                    rewarder=rew,
                    nonadaptive=nona,
                    scale=scale,
                    train=False)
        
    model = PPO.load("mods/cifarbags_" + str(num) + "_model.pt", seed=seed)
    mean_reward, std_reward, epsilons, mean_eps, start_eps, iters, mean_length, _ = evaluate_policy(model, envv, n_eval_episodes=100)
    
    res = [mean_reward, std_reward, mean_eps, start_eps, mean_length]
    c = np.mean([i[1000] for i in epsilons])
    d = np.mean([i[2000] for i in epsilons])
    print('bags0:', res, c, d)
        
    return mean_eps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MA BAGS")
    parser.add_argument('--timesteps', default=int(4e6), type=int, help="Total number of timesteps to run for")
    parser.add_argument('--steps', default=int(5e3), type=int, help="Number of steps for each attack episode")
    parser.add_argument('--adaptive', default=int(2), type=int, help="Controls which agents are adaptive")
    parser.add_argument('--ratio', default=float(0.5), type=float, help="Probability of next draw being benign")
    parser.add_argument('--defended', default=bool(False), type=bool, help="Adversarially trained model or not")
    parser.add_argument('--seed', default=int(2), type=int, help="Seed for all PRNG sources")
    parser.add_argument('--train', default=bool(False), type=bool, help="Train or Test")
    parser.add_argument('--load', default=str("16"), type=str, help="Model to load")
    parser.add_argument('--rew', default=int(3), type=bool, help="Reward used")
    parser.add_argument('--scale', default=int(10), type=int, help="Scale used")
    args = parser.parse_args()
    if args.train:
        # Create a new optuna study.
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, gc_after_trial=True)
    else:
        mean_eps = test(args.load, args.rew, args.scale)
    # train(args)
