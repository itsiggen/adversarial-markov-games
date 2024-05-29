import argparse
import gym
import os
import numpy as np
import optuna
import gc
from tqdm import tqdm
from agents.rppo import RPPO
from agents.benign import RandomAgent
from torchvision import datasets, transforms
from utils.evaluation import evaluate_rdpolicy
from envs.hsja_games import HsjaGames
os.environ['CUDA_VISIBLE_DEVICES'] = ''

transform=transforms.ToTensor()
dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)

def objective(trial):
    """
    AA-AD: Adaptive Adversary - Adaptive Defense
    """
    
    print('Training HSJA-7: AA-AD..')
    
    eval_steps = 5000
    adaptive = 3 # both adaptive 
    ratio = 0.5
    cont = 2 # contrastive model used
    defended = True
    seed = 2

    steps = trial.suggest_categorical('steps', [1000,2500,5000])
    lra = trial.suggest_categorical('lra', [0.003,0.001,0.0001])
    lri = trial.suggest_categorical('lri', [0.003,0.001,0.0001])
    buffer = 1024
    batch = 64
    epochs = 20
    gamma = 0.99
    ent_coef = 0
    vf_coef = 0.5
    rint = trial.suggest_categorical('rint', [2,4,5])
    radv = trial.suggest_categorical('radv', [1,3,5])
    inter = 1
        
    # Create environment
    env = gym.make("HsjaGames-v0",
                   steps=steps,
                   ratio_benign=ratio,
                   adaptive=adaptive,
                   dataset=dataset,
                   train=True,
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
                learning_rate=lri,
                gamma=round(gamma,2),
                tensorboard_log=None,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                verbose=0,
                seed=seed,
                policy_kwargs=dict(net_arch=[dict(vf=[32,32], pi=[32,32])]))

    adversary = RPPO(policy="MlpPolicy",
                env=env,
                agent='adversary',
                n_steps=buffer,
                batch_size=batch,
                n_epochs=epochs,
                learning_rate=lra,
                gamma=round(gamma,2),
                tensorboard_log=None,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                verbose=0,
                seed=seed,
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
        
    for timestep in tqdm(range(total_timesteps), disable=False):
        # Check if a rollout buffer has been filled and train
        check_full(agents)
        # Store previous move
        prev = curr
        # next agent moves
        obs, reward, done, info = agents[nxt].move()
        curr = info["curr"]
        nxt = info["next"]
        n_steps += 1

        if curr == 0:
            if nxt == 0:
                agents[0].proceed(obs, reward, done, info)
            elif nxt == 1:
                if rst == 1:
                    # First time adv plays after start of episode, set first obs
                    agents[1].set_last(obs, False)
                    rst = 0
                else:
                    agents[1].proceed(obs, reward, done, info)
        elif curr == 1 or curr == 2:
            if done:
                agents[0].proceed(obs, reward, False, info)
                done, curr, nxt, n_steps = reset()
                rst = 1
            else:
                agents[0].proceed(obs, reward, done, info)

    # Save the trained agents
    
    print("Saving models...")
    interceptor.save("mods/games/hsjaa7int_" + str(trial.number) + ".pt")
    adversary.save("mods/games/hsjaa7adv_" + str(trial.number) + ".pt")

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

    interceptor = RPPO.load("mods/games/hsjaa7int_" + str(trial.number) + ".pt" , envv, "interceptor", seed)
    adversary = RPPO.load("mods/games/hsjaa7adv_" + str(trial.number) + ".pt", envv, "adversary", seed)
                
    benign = RandomAgent(env=envv)

    mean_rint, std_rint, mean_radv, std_radv, epsilons, lengths, mean_eps, start_eps, mean_acc = evaluate_rdpolicy(interceptor, adversary, benign, envv, n_eval_episodes=15)
    
    res = np.asarray([round(mean_rint,2), round(std_rint,2), round(mean_radv,2), round(std_radv,2), round(start_eps,3), round(mean_eps,3), round(mean_acc,3)])
    print('hsja7:', res)
    
    del env
    del envv
    del interceptor
    del adversary
    del benign
    gc.collect()
    
    return mean_eps, mean_acc
    
def check_full(agents):
    for i in range(2):
        if agents[i].rollout_buffer.full:
            agents[i].close_buffer()
            agents[i].train()
            agents[i].reset_buffer()

def reset():
    return False, 1, 0, 0

def test(num, r1, r2):
    eval_steps = 5000
    adaptive = 3 # both adaptive 
    ratio = 0.5
    defended = False
    cont = 2
    seed = 2
    thres = 3

    # Make evaluation env
    envv = gym.make("HsjaGames-v0",
                    steps=eval_steps,
                    ratio_benign=ratio,
                    adaptive=adaptive,
                    dataset=dataset,
                    defended=defended,
                    cont=cont,
                    train=False,
                    rint=r1,
                    radv=r2)

    interceptor = RPPO.load("mods/games/hsjaa7int_" + str(num) + ".pt", envv, "interceptor", seed)
    adversary = RPPO.load("mods/games/hsjaa7adv_" + str(num) + ".pt", envv, "adversary", seed)

    benign = RandomAgent(env=envv)
    
    mean_rint, std_rint, mean_radv, std_radv, epsilons, iters, mean_eps, start_eps, mean_acc = evaluate_rdpolicy(interceptor, adversary, benign, envv, n_eval_episodes=100)

    # attack success rate
    lsc = [i[-1] for i in epsilons]
    suc = np.sum(np.array(lsc) < thres)
    asr = suc / len(lsc)
    res = [mean_eps, start_eps, mean_acc, asr]
    z = list(zip(iters,epsilons))
    a = [np.interp(1000, i[0], i[1]) for i in z]
    b = [np.interp(2000, i[0], i[1]) for i in z]
    c = np.mean(a)
    d = np.mean(b)
    print(f'hsja7 | defended {defended}:', res, c, d)
        
        
    return mean_eps, mean_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MA BAGS")
    parser.add_argument('--defended', default=bool(False), type=bool, help="Adversarially trained model or not")
    parser.add_argument('--train', default=bool(False), type=bool, help="Train or Test")
    parser.add_argument('--num', default=str("10"), type=str, help="Agents to load")
    parser.add_argument('--r1', default=int(5), type=bool, help="Int reward used")
    parser.add_argument('--r2', default=int(1), type=bool, help="Adv reward used")
    args = parser.parse_args()
    if args.train:
        # Create a new optuna study.
        study = optuna.create_study(directions=['minimize', 'maximize'])
        study.optimize(objective, n_trials=30, gc_after_trial=True)
    else:
        mean_eps, mean_acc = test(args.num, args.r1, args.r2)