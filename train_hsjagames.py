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
from envs.hsja_games import HsjaGames
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

transform=transforms.ToTensor()
dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)
    
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
    
    """
    Best is Trial 10: [7.188713073730469, 0.9932314562122941] and parameters: {'lr': 5e-05, 'buffer': 512, 'batch': 64,
    'epoch': 30, 'gamma': 0.99, 'ent_coef': 0.001, 'vf_coef': 0.5, 'intercept': 1.5, 'rint': 4}. 
    """
    
    timesteps = int(6e5)
    steps = 2150
    eval_steps = 5000
    adaptive = 2
    ratio = 0.5
    defended = False
    seed = 2
    logdir = "./logs/"
    
    lr = trial.suggest_categorical('lr', [0.001,0.0005,0.0001,0.00005])
    buffer = trial.suggest_categorical('buffer', [512,1024,2048,4096])
    batch = trial.suggest_categorical('batch', [64,128])
    epochs = trial.suggest_categorical('epochs', [10,20,30])
    gamma = trial.suggest_float('gamma', 0.7, 0.99, step=0.01)
    ent_coef = trial.suggest_categorical('ent_coef', [0,0.001,0.0001])
    vf_coef = trial.suggest_categorical('vf_coef', [0,0.1,0.5])
    inter = trial.suggest_categorical('intercept', [1,1.5,2])
    # inter = 1
    # radv = trial.suggest_categorical('radv', [1,2,4,5])
    rint = trial.suggest_categorical('rint', [2,4,5])
    radv = 1
    # rint = 2
    
    
    # Create environment
    env = gym.make("HsjaGames-v0", steps=steps, ratio_benign=ratio, adaptive=adaptive, dataset=dataset, rint=rint, radv=radv, defended=defended, intercept=inter)
    total_timesteps = timesteps
    
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
    
    # interceptor = RPPO.load("mods/games/hsjagamesint_best.pt", env, "interceptor",
    #                         seed=seed,
    #                         n_steps=buffer,
    #                         batch_size=batch,
    #                         learning_rate=lr, # 0.00039
    #                         gamma=round(gamma,2), # 0.92
    #                         vf_coef = vf_coef,
    #                         ent_coef = ent_coef, # 0.0001
    #                         )
    # adversary = RPPO.load("mods/games/hsjagamesadv_best.pt", env, "adversary", seed)
    
    # print(interceptor.__dict__)
    
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
        # print("cadence", prev, curr, nxt, reward)

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
    interceptor.save("mods/games/hsjagamesint_" + str(trial.number) + ".pt")
    adversary.save("mods/games/hsjagamesadv_" + str(trial.number) + ".pt")
    
    # Make evaluation env
    envv = gym.make("HsjaGames-v0", steps=eval_steps, ratio_benign=ratio, adaptive=adaptive, dataset=dataset, defended=defended, train=False, rint=rint, radv=radv, intercept=inter)
    # Load the trained agents
    # " + str(trial.number) + "
    interceptor = RPPO.load("mods/games/hsjagamesint_" + str(trial.number) + ".pt", envv, "interceptor", seed)
    adversary = RPPO.load("mods/games/hsjagamesadv_" + str(trial.number) + ".pt", envv, "adversary", seed)
    # interceptor = RPPO.load("mods/games/hsjagamesint_1.pt", envv, "interceptor")
    # adversary = RPPO.load("mods/games/hsjagamesadv_1.pt", envv, "adversary")
    benign = RandomAgent(env=envv)

    mean_rint, std_rint, mean_radv, std_radv, epsilons, lengths, mean_eps, start_eps, mean_acc = evaluate_rdpolicy(interceptor, adversary, benign, envv, n_eval_episodes=20)
    
    res = np.asarray([round(mean_rint,2), round(std_rint,2), round(mean_radv,2), round(std_radv,2), round(mean_eps,3), round(start_eps,3), round(mean_acc,3)])
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
    # Create a new optuna study.
    study = optuna.create_study(directions=['maximize', 'maximize'])
    study.optimize(objective, n_trials=20)
    # train(args)