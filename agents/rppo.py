from typing import Any, Callable, Dict, Optional, Type, Union, Tuple
import io
import pathlib
import numpy as np
import torch as th
import gym
import time
from copy import deepcopy
from .dummy_rvec_env import DummyRvecEnv
from .normalize import RecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common import logger, utils

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.vec_env import VecEnv

class RPPO(PPO):
    """
    Recursive Proximal Policy Optimization algorithm (RPPO)
    
    Adapted from Stable Baselines 3 (https://github.com/DLR-RM/stable-baselines3)
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        agent: str = "interceptor",
        learning_rate: Union[float, Callable] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        mode: int = 0,
        normalize: bool = False,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        self.mode = mode # 0/1 = stochastic/deterministic actions
        self.agent = agent
        env.observation_space = env.observation_spaces[agent]
        env.action_space = env.action_spaces[agent]
        super(RPPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=_init_setup_model,
        )
        
        # Define observation and action space for each type of agent
        self.observation_space = env.observation_spaces[agent]
        self.action_space = env.action_spaces[agent]
        self.normalize = normalize
        if normalize:
            self.RecNorm = RecNormalize(self.observation_space, training=True)
         
    def reset_buffer(self):
        self.n_steps = 0
        self.rollout_buffer.reset()
        
    def close_buffer(self):
        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(np.array(self._last_obs)).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)
        self.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=self._last_dones)
        
    def move(self):
        assert self._last_obs is not None, "No previous observation was provided"
        # Sample new weights for the state dependent exploration

        if self.use_sde and self.sde_sample_freq > 0 and self.n_steps % self.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.policy.reset_noise(self.env.num_envs)

        with th.no_grad():
            # Convert to pytorch tensor
            obs_tensor = th.as_tensor(np.array(self._last_obs)).to(self.device)
            if self.mode:
                # vectorized, remove first dim to pass to predict
                obs_tensor = th.squeeze(obs_tensor, 0)
                actions, _ = self.policy.predict(obs_tensor.cpu(), deterministic=True)
                actions = np.expand_dims(actions, axis=0)
            else:
                actions, self.values, self.log_probs = self.policy.forward(obs_tensor)
                actions = actions.cpu().numpy()

        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Check consequences of returning clipped actions instead of actions
        self.actions = actions
        # print(self.actions, self.agent)
        new_obs, reward, done, info = self.env.step(clipped_actions)
        
        return new_obs, reward, done, info
               
    def proceed(self, new_obs, rewards, dones, infos):
        self.num_timesteps += self.env.num_envs

        # Normalize obs & reward
        if self.normalize:
            new_obs, rewards = self.RecNorm.norm(new_obs, rewards)

        # Update dummyvecenv buffers
        self.env.step_proceed(new_obs, rewards, dones, infos)

        self._update_info_buffer([infos])
        self.n_steps += 1
        # print(self.agent)
        # Check if rollout buffer is filled with the correct obs/rewards etc
        if not self.mode:
            self.rollout_buffer.add(self._last_obs, self.actions, rewards, self._last_dones, self.values, self.log_probs)
            # print(self._last_obs, self.actions, rewards, self._last_dones, self.values)
        self._last_obs = [new_obs]
        self._last_dones = dones
        # print(self.agent, self.env.envs[0].past, self.env.envs[0].curr, self.env.envs[0].next, rewards)
        
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        if self.normalize:
            observation = self.RecNorm.normalize_obs(observation)
        return self.policy.predict(observation, state, mask, deterministic)
    
    def set_last(self, obs, done):
        self._last_obs = [obs]
        self._last_dones = done
    
    def setup_learn(self, reset_num_timesteps: bool = True, tb_log_name: str = "run"):
        """
        Initialize different variables needed for training.
        :param total_timesteps: The total number of samples (env steps) to train on
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :return:
        """
        self.start_time = time.time()

        if self.action_noise is not None:
            self.action_noise.reset()

            # if self._vec_normalize_env is not None:
            #     self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # Configure logger's outputs
        utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

    def _wrap_env(self, env: GymEnv, verbose: int = 0) -> VecEnv:
        if not isinstance(env, VecEnv):
            # print("Wrapping the env in a DummyRVecEnv.")
            env = DummyRvecEnv([lambda: env], self.agent)
        return env
    
    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: [GymEnv] = None,
        agent: str = "interceptor",
        seed: Optional[int] = None,
        normed: Optional[str] = None,
        device: Union[th.device, str] = "auto",
        **kwargs):
        """
        Load the model from a zip-file

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param kwargs: extra arguments to change the model when loading
        """
        data, params, pytorch_variables = load_from_zip_file(path, device=device)

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # if env is not None:
        #     # Wrap first if needed
        #     env = cls._wrap_env(env, data["verbose"])
            
        #     # Check if given env is valid
        #     check_for_correct_spaces(env, data["observation_space"], data["action_space"])
        # else:
        #     # Use stored env, if one exists. If not, continue as is (can be used for predict)
        #     if "env" in data:
        #         env = data["env"]

        # noinspection PyArgumentList
        model = cls(
            policy=data["policy_class"],
            env=env,
            agent=agent,
            seed=seed,
            device=device,
            mode=1,
            _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        # set seed and mode
        model.seed = seed
        model.mode = 1
        model._setup_model()

        # put state_dicts back in place
        model.set_parameters(params, exact_match=True, device=device)
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                recursive_setattr(model, name, pytorch_variables[name])

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # pytype: disable=attribute-error
            
        # Load normalization object:
        if normed:
            model.normalize=True
            model.RecNorm = RecNormalize.load(normed)
            model.RecNorm.training=False
            model.RecNorm.norm_reward=False
 
        return model
