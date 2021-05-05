from typing import Callable, List, Optional, Tuple, Union

import gym
import numpy as np

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

    episode_rewards, episode_lengths, epsilons, acc = [], [], [], []
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
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
            epsilon.append(_info['epsilon'])
            correct = _info['correct']
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        epsilons.append(epsilon)
        acc.append(correct)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_acc = np.mean(acc)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward, epsilons, mean_acc

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

    episode_rewards, lengths, epsilons, acc = [], [], [], []
    
    agents = [interceptor, adversary, benign]
    
    # Reset once as VecEnv are reset automatically afterwards
    obs = env.reset()
    for i in range(n_eval_episodes):
        done, state = False, None
        curr, nxt = 1, 0
        agent_rewards = [0.0,0.0,0.0]
        agent_steps = [0,0,0]
        epsilon = []
        while not done:
            prev = curr
            action, state = agents[nxt].predict(obs, deterministic=deterministic)
            # loading the model returns an extra dim, hence squeeze
            obs, reward, done, _info, curr, nxt = env.step(action)
            # print(curr, nxt)
            agent_rewards[prev] += reward
            # if curr == 0:
            #     elif curr = 1:
            #         else:
            agent_steps[curr] += 1
            if curr == 0 and prev == 1:
                epsilon.append(_info['epsilon'])
                correct = _info['correct']
        episode_rewards.append(agent_rewards)
        lengths.append(agent_steps)
        epsilons.append(epsilon)
        acc.append(correct)
    mean_reward_int = np.mean(episode_rewards[0])
    std_reward_int = np.std(episode_rewards[0])
    mean_reward_adv = np.mean(episode_rewards[1])
    std_reward_adv = np.std(episode_rewards[1])
    mean_acc = np.mean(acc)

    return mean_reward_int, std_reward_int, mean_reward_adv, std_reward_adv, epsilons, mean_acc