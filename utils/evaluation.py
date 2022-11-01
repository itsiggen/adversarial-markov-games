from typing import Callable, List, Optional, Tuple, Union

import gym
import numpy as np
from tqdm import tqdm

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv


def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of reward per episode
        will be returned instead of the mean.
    :return: Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths, iterations, epsilons, start, acc = [], [], [], [], [], []
    for i in tqdm(range(n_eval_episodes), disable=False):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        iters = []
        epsilon = []
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            # print(action)
            # loading the model returns an extra dim, hence squeeze
            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            iters.append(_info['iters'])
            epsilon.append(_info['epsilon'])
            gap = _info['gap']
            correct = _info['correct']
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        iterations.append(iters)
        epsilons.append(epsilon)
        # print('iter:', i, len(epsilon))
        start.append(gap)
        acc.append(correct)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_acc = np.mean(acc)
    mean_eps = np.mean([i[-1] for i in epsilons])
    start_eps = np.mean(start)
    mean_length = np.mean(episode_lengths)
    
    return mean_reward, std_reward, epsilons, mean_eps, start_eps, iterations, mean_length, mean_acc

def evaluate_rpolicy(
    interceptor,
    adversary,
    benign,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    callback: Optional[Callable] = None,
    reward_threshold: Optional[float] = None,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Similar to evaluate_policy, but for competitive envs and returns average reward.
    This is made to work only with one env.

    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, lengths, epsilons, start, acc = [], [], [], [], []
    
    agents = [interceptor, adversary, benign]

    for i in tqdm(range(n_eval_episodes), disable=False):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        curr, nxt = 1, 0
        agent_rewards = [0.0,0.0,0.0]
        agent_steps = [0,0,0]
        epsilon = []
        while not done:
            prev = curr
            action, state = agents[nxt].predict(obs, deterministic=deterministic)
            obs, reward, done, _info, = env.step(action)
            curr = _info["curr"]
            nxt = _info["next"]
            # print(curr, nxt)
            agent_rewards[prev] += reward
            # print(agent_rewards)
            # if curr == 0:
            #     elif curr = 1:
            #         else:
            agent_steps[curr] += 1
            if curr == 0 and prev == 1:
                epsilon.append(_info['epsilon'])
                # correct = _info['correct']
            if done:
                gap = _info['gap']
                correct = _info['correct']
                curr, nxt = 1, 0
        episode_rewards.append(agent_rewards)
        lengths.append(agent_steps)
        epsilons.append(epsilon)
        start.append(gap)
        acc.append(correct)
    mean_reward_int = np.mean([i[0] for i in episode_rewards]) #WRONG -> [i[0] for i in episode_rewards]
    std_reward_int = np.std([i[0] for i in episode_rewards])
    mean_reward_adv = np.mean([i[1] for i in episode_rewards])
    std_reward_adv = np.std([i[1] for i in episode_rewards])
    mean_acc = np.mean(acc)
    mean_eps = np.mean([i[-1] for i in epsilons])
    start_eps = np.mean(start)

    return mean_reward_int, std_reward_int, mean_reward_adv, std_reward_adv, epsilons, lengths, mean_eps, start_eps, mean_acc

def evaluate_rdpolicy(
    interceptor,
    adversary,
    benign,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    callback: Optional[Callable] = None,
    reward_threshold: Optional[float] = None,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Similar to evaluate_policy, but for competitive envs and returns average reward.
    This is made to work only with one env.
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, lengths, iterations, epsilons, start, acc = [], [], [], [], [], []
    
    agents = [interceptor, adversary, benign]

    for i in tqdm(range(n_eval_episodes), disable=False):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        curr, nxt = 1, 0
        rst = 1
        agent_rewards = [0.0,0.0,0.0]
        agent_steps = [0,0,0]
        epsilon = []
        iters = []
        while not done:
            prev = curr
            action, state = agents[nxt].predict(obs, deterministic=deterministic)
            # action = action[0]
            # print(action)
            # loading the model returns an extra dim, hence squeeze
            obs, reward, done, _info = env.step(action)
            curr = _info["curr"]
            nxt = _info["next"]
            # print(curr, nxt)
            if curr == 0:
                if nxt == 0:
                    agent_rewards[0] += reward
                    # if reward != 0.2 and reward != 0: print("rew:", reward, _info['iterations'], _info['epsilon'], env.imp)
                    # print("rew:", reward, _info['iterations'], _info['epsilon'], env.imp) 
                    
                elif nxt == 1:
                    if rst == 1:
                        agents[1].set_last(obs, False)
                        rst = 0
                    else:
                        agent_rewards[1] += reward
            elif curr == 1 or curr == 2:
                agent_rewards[0] += reward
                # if reward != 0.2 and reward != 0: print("rew:", reward, _info['iterations'], _info['epsilon'], env.imp)
                # print("rew:", reward, _info['iterations'], _info['epsilon'], env.imp) 
                
            agent_steps[curr] += 1
            if curr == 0 and prev == 1:
                epsilon.append(_info['epsilon'])
                iters.append(_info['iterations'])
                # correct = _info['correct']
                # print(env.iter, done)
            if done:
                # print(done)
                # print('DONE', epsilon[-1])
                gap = _info['gap']
                correct = _info['correct']
                curr, nxt = 1, 0
        episode_rewards.append(agent_rewards)
        lengths.append(agent_steps)
        epsilons.append(epsilon)
        iterations.append(iters)
        start.append(gap)
        # print(epsilon[-1] - gap)
        acc.append(correct)
        # print(i)
    mean_reward_int = np.mean([i[0] for i in episode_rewards]) #WRONG -> [i[0] for i in episode_rewards]
    std_reward_int = np.std([i[0] for i in episode_rewards])
    mean_reward_adv = np.mean([i[1] for i in episode_rewards])
    std_reward_adv = np.std([i[1] for i in episode_rewards])
    mean_acc = np.mean(acc)
    mean_eps = np.mean([i[-1] for i in epsilons])
    start_eps = np.mean(start)
    # print(len(epsilons), mean_eps, start_eps)
    # print([i[-1] for i in epsilons], start)

    return mean_reward_int, std_reward_int, mean_reward_adv, std_reward_adv, epsilons, iterations, mean_eps, start_eps, mean_acc

def evaluate_rtpolicy(
    interceptor,
    adversary,
    benign,
    env: Union[gym.Env, VecEnv],
    act_size: int = 3,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    callback: Optional[Callable] = None,
    reward_threshold: Optional[float] = None,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Similar to evaluate_policy, but with dummy moves
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, lengths, iterations, epsilons, start, acc = [], [], [], [], [], []

    for i in tqdm(range(n_eval_episodes), disable=False):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        curr, nxt = 1, 0
        # rst = 1
        # agent_steps = [0,0,0]
        epsilon = []
        iters = []
        while not done:
            prev = curr
            # action, state = agents[nxt].predict(obs, deterministic=deterministic)
            # action = action[0]
            # print(action)
            # loading the model returns an extra dim, hence squeeze
            if nxt == 0:
                action = np.zeros(shape=(1,))
            else:
                action = np.zeros(shape=(act_size,))
            
            obs, reward, done, _info = env.step(action)
            curr = _info["curr"]
            nxt = _info["next"]

            if curr == 0 and prev == 1:
                epsilon.append(_info['epsilon'])
                iters.append(_info['iterations'])
                # correct = _info['correct']
                # print(correct)
                # print(env.iter, done)
            if done:
                # print(done)
                # print('DONE', epsilon[-1])
                gap = _info['gap']
                correct = _info['correct']
                # print(correct)
                curr, nxt = 1, 0
        epsilons.append(epsilon)
        iterations.append(iters)
        start.append(gap)
        # print(epsilon[-1] - gap)
        acc.append(correct)
        # print(i)
    mean_reward_int = 0
    std_reward_int = 0
    mean_reward_adv = 0
    std_reward_adv = 0
    mean_acc = np.mean(acc)
    mean_eps = np.mean([i[-1] for i in epsilons])
    start_eps = np.mean(start)
    # print(len(epsilons), mean_eps, start_eps)
    # print([i[-1] for i in epsilons], start)

    return mean_reward_int, std_reward_int, mean_reward_adv, std_reward_adv, epsilons, iterations, mean_eps, start_eps, mean_acc