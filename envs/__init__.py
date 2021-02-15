from gym.envs.registration import register
import gym

# envs = gym.envs.registry.env_specs.copy()
# for env in envs:
#     if 'BoundaryStep-v0' not in env:
register(
    id='BoundaryStep-v0',
    entry_point='envs.boundary_step:BoundaryStep'
    )