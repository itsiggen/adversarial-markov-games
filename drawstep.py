''' An example of learning a Deep-Q Agent on Leduc Holdem
'''
import torch
import os

import rlcard
from rlcard.agents import DQNAgentPytorch as DQNAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

# Make environment
env = rlcard.make('leduc-holdem', config={'seed': 0})
eval_env = rlcard.make('leduc-holdem', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate the performance
evaluate_every = 100
evaluate_num = 1000
episode_num = 100000

# The intial memory size
memory_init_size = 1000

# Train the agent every X steps
train_every = 1

# The paths for saving the logs and learning curves
log_dir = './experiments/limit_holdem_dqn_result/'

# Set a global seed
set_global_seed(0)

adversary = DQNAgent(scope='dqn',
                 action_num=env.action_num,
                 replay_memory_init_size=memory_init_size,
                 train_every=train_every,
                 state_shape=env.state_shape,
                 mlp_layers=[128, 128],
                 device=torch.device('cpu'))

interceptor = PPO(policy="MlpPolicy",
            env=env,
            learning_rate=lr,
            gamma=gamma,
            tensorboard_log=None,
            ent_coef = ent_coef,
            use_sde=sde,
            verbose=0,
            seed=2,
            policy_kwargs=dict(net_arch=[dict(vf=[64,64], pi=[64,64])]))

adversary = PPO(policy="MlpPolicy",
            env=env,
            learning_rate=lr,
            gamma=gamma,
            tensorboard_log=None,
            ent_coef = ent_coef,
            use_sde=sde,
            verbose=0,
            seed=2,
            policy_kwargs=dict(net_arch=[dict(vf=[64,64], pi=[64,64])]))

extraneous = RandomAgent(action_num=eval_env.action_num)

env.set_agents([agent, random_agent])
eval_env.set_agents([agent, random_agent])

for episode in range(episode_num):
    
    # Reset env and agents, and get the first move
    for i in agents:
        agents[i].reset_episode()
    self.obs, self.reward, self.done, self.info, self.curr, self.next = self.reset()
    n_steps = 0

    # Loop till end of episode
    while not done:
        # Store previous move
        prev = [self.obs, self.reward, self.done, self.info, self.curr, self.next]
        # Next agent moves
        self.obs, self.reward, self.done, self.info, self.curr, self.next = agents[self.next].play()
        n_steps += 1
        # Check which agent played before last move
        if prev[4] == 0:
             agents[0].proceed(*prev[0:3])
        elif prev[4] == 1:
            if n_steps != 0:
                agents[1].proceed(*prev[0:3])
        elif prev[4] == 2:
            agents[2].proceed(*prev[0:3])

        # Break if episode is complete
        if self.done:
            for i in agents:
                agents[i].close_episode()
            break  

    # After episode concludes, train the agents
    for i in agents:
        agents[i].train()
        
# Close files in the logger
logger.close_files()

# Plot the learning curve
logger.plot('DQN')

# Save model
save_dir = 'models/leduc_holdem_dqn_pytorch'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
state_dict = agent.get_state_dict()
print(state_dict. keys())
torch.save(state_dict, os.path.join(save_dir, 'model.pth'))
    

def collect_rollouts(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int
    ) -> bool:

        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            
            # Draw sample
            new_obs, rewards, dones, infos = env.draw(clipped_actions)
            # Decide
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self