from gym.envs.registration import register
import gym
for env in list(gym.envs.registry.env_specs):
     if 'BoundaryStep-v0' in env:
          # print('Remove {} from registry'.format(env))
          del gym.envs.registry.env_specs[env]
for env in list(gym.envs.registry.env_specs):
     if 'BoundarySkip-v0' in env:
          # print('Remove {} from registry'.format(env))
          del gym.envs.registry.env_specs[env]

register(
    id='BoundaryStep-v0',
    entry_point='envs.boundary_step:BoundaryStep'
   )

register(
    id='BoundarySkip-v0',
    entry_point='envs.boundary_skip:BoundarySkip'
    )