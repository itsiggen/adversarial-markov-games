from gym.envs.registration import register
import gym
for env in list(gym.envs.registry.env_specs):
     if 'BoundaryStep-v0' in env:
          del gym.envs.registry.env_specs[env]
for env in list(gym.envs.registry.env_specs):
     if 'BoundarySkip-v0' in env:
          del gym.envs.registry.env_specs[env]
for env in list(gym.envs.registry.env_specs):
     if 'BagsSkip-v0' in env:
          del gym.envs.registry.env_specs[env]
for env in list(gym.envs.registry.env_specs):
     if 'BagsSkipCIFAR-v0' in env:
          del gym.envs.registry.env_specs[env]
for env in list(gym.envs.registry.env_specs):
     if 'HsjaSkip-v0' in env:
          del gym.envs.registry.env_specs[env]
for env in list(gym.envs.registry.env_specs):
     if 'HsjaSkipCIFAR-v0' in env:
          del gym.envs.registry.env_specs[env]
for env in list(gym.envs.registry.env_specs):
     if 'BagsGames-v0' in env:
          del gym.envs.registry.env_specs[env]
for env in list(gym.envs.registry.env_specs):
     if 'BagsGamesCIFAR-v0' in env:
          del gym.envs.registry.env_specs[env]
for env in list(gym.envs.registry.env_specs):
     if 'HsjaGames-v0' in env:
          del gym.envs.registry.env_specs[env]
for env in list(gym.envs.registry.env_specs):
     if 'HsjaGamesCIFAR-v0' in env:
          del gym.envs.registry.env_specs[env]
for env in list(gym.envs.registry.env_specs):
     if 'TestGames-v0' in env:
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
    id='BagsSkipCIFAR-v0',
    entry_point='envs.bags_skip_cifar:BagsSkipCIFAR'
    )

register(
    id='HsjaSkip-v0',
    entry_point='envs.hsja_skip:HsjaSkip'
    )

register(
    id='HsjaSkipCIFAR-v0',
    entry_point='envs.hsja_skip_cifar:HsjaSkipCIFAR'
    )

register(
    id='BagsGames-v0',
    entry_point='envs.bags_games:BagsGames'
    )

register(
    id='BagsGamesCIFAR-v0',
    entry_point='envs.bags_games_cifar:BagsGamesCIFAR'
    )

register(
    id='HsjaGames-v0',
    entry_point='envs.hsja_games:HsjaGames'
    )

register(
    id='HsjaGamesCIFAR-v0',
    entry_point='envs.hsja_games_cifar:HsjaGamesCIFAR'
    )

register(
    id='TestGames-v0',
    entry_point='envs.test_games:TestGames'
    )