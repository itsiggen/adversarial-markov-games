import argparse
import gym
import os
os.environ["CUDA_VISIBLE_DEVICES"]=''
import numpy as np
import optuna
from tqdm.auto import tqdm
from agents.rppo import RPPO
from agents.benign import RandomAgent
from torchvision import datasets, transforms
from utils.evaluation import evaluate_rdpolicy
from envs.bags_blacklight_cifar import BagsBlacklightCIFAR
os.environ['CUDA_VISIBLE_DEVICES'] = ''

transform = transforms.ToTensor()
dataset = datasets.CIFAR10('data', train=False, transform=transform, download=True)
    
def objective(trial):
    """
    OA-AD: OARS Adversary - Adaptive Defense
    """
    print('Training OA-AD..')
    
    eval_steps = 5000
    adaptive = 3
    ratio = 0.5
    stt = 2 # defender is learning
    defended = False
    cont = 1 # contrastive model used
    seed = 2

    steps = trial.suggest_categorical('steps', [1000,2000,3000])
    lr = trial.suggest_categorical('lr', [0.003,0.001,0.0003,0.0001])
    buffer = 2048
    batch = 64
    epochs = 20
    gamma = trial.suggest_float('gamma', 0.8, 0.99, step=0.01)
    ent_coef = 0
    rint = trial.suggest_categorical('rint', [2,3,4,5])
    radv = 1
    ts = trial.suggest_categorical('ts', [6e5,10e5,15e5])
    steps = 1000
    
    # Create environment
    env = gym.make("BagsOARSCIFAR-v0",
                   steps=steps,
                   ratio_benign=ratio,
                   adaptive=adaptive,
                   dataset=dataset,
                   rint=rint,
                   radv=radv,
                   cont=cont,
                   defended=defended)
    
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
                verbose=0,
                seed=seed,
                policy_kwargs=dict(net_arch=[dict(vf=[32,32], pi=[32,32])]))
     
    adversary = RPPO(policy="MlpPolicy",
                env=env,
                agent='adversary',
                n_steps=2048,
                batch_size=batch,
                n_epochs=epochs,
                learning_rate=lr,
                gamma=round(gamma,2),
                tensorboard_log=None,
                ent_coef = ent_coef,
                verbose=0,
                seed=seed,
                mode=2,
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
    rst = 1

    pbar = tqdm(range(total_timesteps), disable=False)
    for timestep in pbar:
        # Check if a rollout buffer has been filled and train
        check_full(agents, stt)
        # Store previous move
        prev = curr
        # next agent moves
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
                agents[0].proceed(obs, reward, False, info)
                done, curr, nxt, n_steps = reset()
            else:
                agents[0].proceed(obs, reward, done, info)
    
    # Save the trained agents
    print("Saving models...")
    interceptor.save("mods/oars/bagsint_" + str(trial.number) + ".pt")
    adversary.save("mods/oars/bagsadv_" + str(trial.number) + ".pt")

    # Make evaluation env
    # Different seed for validation pairs.
    seed = 3
    envv = gym.make("BagsOARSCIFAR-v0",
                    steps=eval_steps,
                    ratio_benign=ratio,
                    adaptive=adaptive,
                    dataset=dataset,
                    defended=defended,
                    cont=cont,
                    rint=rint,
                    radv=radv)
    
    interceptor = RPPO.load("mods/oars/bagsint_" + str(trial.number) + ".pt", envv, "interceptor", seed)
    adversary = RPPO.load("mods/oars/bagsadv_" + str(trial.number) + ".pt", envv, "adversary", seed)
                
    benign = RandomAgent(env=envv)    

    mean_rint, std_rint, mean_radv, std_radv, epsilons, lengths, mean_eps, start_eps, mean_acc = evaluate_rdpolicy(interceptor, adversary, benign, envv, n_eval_episodes=15)
    
    res = np.asarray([round(mean_rint,2), round(std_rint,2), round(mean_radv,2), round(std_radv,2), round(start_eps,3), round(mean_eps,3), round(mean_acc,3)])
    print(res)
    
    del env
    del envv
    del interceptor
    del adversary
    del benign
    
    return mean_eps, mean_acc
    
def check_full(agents, stt):
    for i in range(2):
        if agents[i].rollout_buffer.full:
            agents[i].close_buffer()
            if stt == i or stt == 2:
                agents[i].train()
            agents[i].reset_buffer()

def reset():
    return False, 1, 0, 0

def save_results(study, trial):
    with open("cbags_oars_results.txt", "a") as f:
        f.write(f"Trial {trial.number} has finished with values: {trial.values} and parameters: {trial.params}\n")

def test(num, rew):
    eval_steps = 5000
    adaptive = 0
    ratio = 0.5
    defended = True
    black = True
    cont = 1
    seed = 2
    thres = 3

    # Make evaluation env
    env = gym.make("BagsOARSCIFAR-v0",
                    steps=eval_steps,
                    ratio_benign=ratio,
                    adaptive=adaptive,
                    dataset=dataset,
                    defended=defended,
                    train=False,
                    test=True,
                    black=black,
                    cont=cont,
                    rint=rew,
                    radv=1,)

    interceptor = RPPO.load("mods/oars/bagsint_" + str(num) + ".pt" , env, "interceptor", seed)
    adversary = RPPO.load("mods/oars/bagsadv_" + str(num) + ".pt" , env, "adversary", seed)

    benign = RandomAgent(env=env)

    mean_rint, std_rint, mean_radv, std_radv, epsilons, iters, mean_eps, start_eps, mean_acc = evaluate_rdpolicy(interceptor, adversary, benign, env, n_eval_episodes=100)
    
    # attack succes rate
    lsc = [i[-1] for i in epsilons]
    suc = np.sum(np.array(lsc) < thres)
    asr = suc / len(lsc)
    res = [mean_eps, start_eps, mean_acc, asr]
    c = np.mean([i[1000] for i in epsilons])
    d = np.mean([i[2000] for i in epsilons])
    print(res, c, d)
        
    return mean_eps, mean_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MA BAGS")
    parser.add_argument('--defended', default=bool(False), type=bool, help="Adversarially trained model or not")
    parser.add_argument('--train', default=bool(False), type=bool, help="Train or Test")
    parser.add_argument('--load', default=str("31"), type=str, help="Model to load")
    parser.add_argument('--rew', default=int(8), type=int, help="Reward used")
    args = parser.parse_args()
    if args.train:
        # Create a new optuna study.
        study = optuna.create_study(directions=['maximize', 'maximize'])
        study.optimize(objective, n_trials=30, gc_after_trial=True, callbacks=[save_results])
    else:
        mean_eps, mean_acc = test(args.load, args.rew)