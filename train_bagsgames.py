import argparse
import gym
import pandas as pd
import numpy as np
from agents.rppo import RPPO
from agents.benign import RandomAgent
from torchvision import datasets, transforms
from utils.evaluation import evaluate_rpolicy
from envs.bags_games import BagsGames

def train(args):
    """
    Train and save the adversary and intereceptor agents for BAGS
    :param args: (ArgumentParser) the input arguments
    """
    transform=transforms.ToTensor()
    dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)
    # Create environment
    steps = 1000
    env = gym.make("BagsGames-v0", steps=steps, ratio_benign=0.5, adaptive=2, dataset=dataset, seed=2)
    logdir = "./logs"
    total_timesteps = int(5e5)
    
    interceptor = RPPO(policy="MlpPolicy",
                env=env,
                agent='interceptor',
                n_steps=steps,
                learning_rate=0.00039,
                gamma=0.92,
                tensorboard_log=None,
                ent_coef = 0.0001,
                verbose=0,
                seed=2,
                policy_kwargs=dict(net_arch=[dict(vf=[32,32], pi=[32,32])]))
    
    adversary = RPPO(policy="MlpPolicy",
                env=env,
                agent='adversary',
                n_steps=steps,
                learning_rate=0.00056,
                gamma=0.89,
                tensorboard_log=None,
                # ent_coef = ent_coef,
                verbose=0,
                seed=2,
                policy_kwargs=dict(net_arch=[8,8]))
    
    benign = RandomAgent(env=env)
      
    agents = [interceptor, adversary, benign]
    
    
    for agent in agents:
        agent.setup_learn()
    obs = env.reset()
    agents[0].set_last(obs, False)
    done = False
    curr, nxt = 1, 0
    n_steps = 0
        
    for timestep in range(total_timesteps):
        # Check if a rollout buffer has been filled and train
        check_full(agents)
        # Store previous move
        prev = curr
        # next agent moves
        # print(nxt)
        obs, reward, done, info, curr, nxt = agents[nxt].move()
        n_steps += 1
        
        if curr == 0:
            if n_steps == 1:
                # env has been just reset
                agents[1].set_last(obs, False)
            else:
                agents[prev].proceed(obs, reward, done, info)
        elif curr == 1 or curr == 2:
            if done:
                print(info)
                term_obs = agents[1].env.get_obs()
                agents[0].proceed(term_obs, reward, done, info)
                done, curr, nxt, n_steps = reset()
            else:
                agents[0].proceed(obs, reward, done, info)

    # Save the trained agents
    print("Saving models...")
    interceptor.save("interceptor_model")
    adversary.save("adversary_model")
    
    # Make evaluation env
    envv = gym.make("BagsGames-v0", steps=steps, ratio_benign=0.5, adaptive=2, dataset=dataset, train=False, seed=2)
    # Load the trained agents
    interceptor = RPPO.load("interceptor_model", envv, "interceptor")
    adversary = RPPO.load("adversary_model", envv, "adversary")
    benign = RandomAgent(env=envv)

    mean_rint, std_rint, mean_radv, std_radv, epsilons, mean_acc = evaluate_rpolicy(interceptor, adversary, benign, envv, n_eval_episodes=2)
    
    # res = np.asarray([mean_reward, std_reward, mean_epsilon, mean_acc])
    # np.savetxt('./logs/50benign.csv', res, delimiter=";", fmt='%1.3f')
    # print(res)
    
    # df = pd.DataFrame({'mean_reward': mean_reward, 'std_reward': std_reward, 'mean_epsilon': mean_epsilon, 'mean_acc': mean_acc})
    # # file_path = os.path.join(logdir, '50benign.csv')
    # df.to_csv('/logs/50benign.csv', index=Falsef, loat_format='%.3f')
    
def check_full(agents):
    for i in range(2):
        if agents[i].rollout_buffer.full:
            print("Training agent...")
            agents[i].close_buffer()
            agents[i].train()
            agents[i].reset_buffer()

def reset():
    return False, 1, 0, 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MA BAGS")
    parser.add_argument('--episodes', default=int(1e3), type=int, help="Maximum number of episodes")
    args = parser.parse_args()
    train(args)
    
        
    # for episode in range(total_episodes):
    
    #     # Reset env and agents, and get the first move
    #     for agent in agents:
    #         agent.reset_episode()
    #     obs = env.reset()
    #     agents[0].set_last(obs, False)
         
    #     done = False
    #     curr = 1
    #     nxt = 0
    #     n_steps = 0
    
    #     # Loop till end of episode
    #     while not done:
    #         # Store previous move
    #         # prev = [obs, reward, done, info, curr, nxt]
    #         prev = curr
    #         # next agent moves
    #         # print(nxt)
    #         obs, reward, done, info, curr, nxt = agents[nxt].move()
    #         n_steps += 1

    #         if curr == 0:
    #             if n_steps == 1:
    #                 # Pass correct last_obs after reset to adversary
    #                 agents[1].set_last(obs, False)
    #             elif done:
    #                 print('done2')
    #                 agents[1].proceed(obs, reward, done, info)
    #                 agents[1].close_episode(n_steps - steps)
    #             else:
    #                 agents[prev].proceed(obs, reward, done, info)
    #         elif curr == 1 or curr == 2:
    #             if done:
    #                 print('done1')
    #                 # obs = env.get_obs()
    #                 agents[0].proceed(obs, reward, done, info)
    #                 agents[0].close_episode(n_steps - steps)
    #                 done = False
    #             else:
    #                 agents[0].proceed(obs, reward, done, info)
            
    #         # Break if episode is complete
    #         # if done:
    #         #     for i in range(len(agents)):
    #         #         agents[i].close_episode()
    #         #     break   
            
    #     # After episode concludes, train the agents
    #     for i in range(len(agents)):
    #         agents[i].train()