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
for env in list(gym.envs.registry.env_specs):
     if 'BagsSkip-v0' in env:
          # print('Remove {} from registry'.format(env))
          del gym.envs.registry.env_specs[env]
for env in list(gym.envs.registry.env_specs):
     if 'HsjaSkip-v0' in env:
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

register(
    id='BagsSkip-v0',
    entry_point='envs.bags_skip:BagsSkip'
    )

register(
    id='HsjaSkip-v0',
    entry_point='envs.hsja_skip:HsjaSkip'
    )