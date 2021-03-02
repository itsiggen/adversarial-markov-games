import eagerpy as ep
import numpy as np
import gym
import torch
import random
from torchvision import datasets, transforms
from foolbox import PyTorchModel
from foolbox.tensorboard import TensorBoard
from foolbox.attacks import BoundaryAttack
from foolbox.attacks.boundary_attack import ArrayQueue, draw_proposals
from foolbox.attacks.base import get_is_adversarial
from foolbox.distances import l2
from gym import error, spaces, utils
from foolbox.criteria import TargetedMisclassification
from utils.utils import flatten, atleast_kd
from models.trainMNISTtorch import Net
from joblib import load
from collections import deque
import matplotlib.pyplot as plt
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BoundarySkip(gym.Env):
    def __init__(
        self,
        steps: int = 1000,
        spherical_step: float = 2e-2,
        source_step: float = 2e-2,
        source_step_convergence: float = 1e-7,
        step_adaptation: float = 1.5,
        nonadaptive = False,
        train = True,
        tensorboard = False,
        update_stats_every_k: int = 10
        ):
        super(BoundarySkip, self).__init__()
        # random.seed(steps)        
        
        # Boundary Attack inits
        self.steps = steps
        self.spherical_step = spherical_step
        self.source_step = source_step
        self.source_step_convergence = source_step_convergence
        self.step_adaptation = step_adaptation
        self.nonadaptive = nonadaptive
        self.tensorboard = tensorboard
        self.update_stats_every_k = update_stats_every_k
        
        # Actions controlled by the adversary
        # self.action_space = spaces.Box(low=1e-5, high=1 - 1e-5, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32)
        # Observation space is the MNIST inputs
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

        # Load MNIST pytorch CNN model -- 99.1% acc
        transform=transforms.ToTensor()
        self.dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)
        self.mode = Net()
        self.mode.load_state_dict(torch.load('models/mnist_cnn.pt'))
        self.mode.eval()
        self.model = PyTorchModel(self.mode, bounds=(0, 1))
        self.indices = [0,7999] if train else [8000,9999]
    
        self.done = False
        
    def reset(self):       
        # Initialize new targeted attack 
        self.tb = TensorBoard(logdir=self.tensorboard)
        self.starting_point, startLabel, self.wanted_point, originLabel = self.get_pair()
        self.starting_point, _ = ep.astensor_(self.starting_point)
        print("Start:", startLabel, "| Wanted:", originLabel)
        self.criterion = TargetedMisclassification(torch.tensor([startLabel]))
        self.originals, self.restore_type = ep.astensor_(self.wanted_point)
        # Initial distance between starting and origin points
        self.gap = l2(self.starting_point, self.wanted_point).numpy()
        self.dist = self.gap
        # Distance between successive steps
        self.diff = np.float32(0.0)
        # Moving average of the closing distance
        self.dist_moving = np.float32(1.0)
        self.iter = 0
        # Last iter for improvement
        self.improve_last = 0
        self.improve_avg = 0
        # Moving average of the step
        self.step_moving = 0.1
        self.gain_moving = 0.1
        slope = 0.5
        # Target epsilon
        self.epsilon = 1
        self.actions = []
        self.done = False
        self.success = False

        self.is_adversarial = get_is_adversarial(self.criterion, self.model)
        
        if self.starting_point is None:
            raise ValueError("no starting_point provided")
        else:
            self.best_advs = ep.astensor(self.starting_point)

        is_adv  = self.is_adversarial(self.best_advs.raw.unsqueeze(1))
        if not is_adv:
            raise ValueError("starting_point is not adversarial")

        self.N = len(self.originals) # must be 1 as we perform 1 attack at a time
        self.ndim = self.originals.ndim
        self.bounds = self.model.bounds
        self.spherical_steps = ep.ones(self.originals, self.N) * self.spherical_step
        self.source_steps = ep.ones(self.originals, self.N) * self.source_step
        self.unnormalized_source_directions = self.originals - self.best_advs
        self.source_norms = ep.norms.l2(flatten(self.unnormalized_source_directions), axis=-1)
        self.source_directions = self.unnormalized_source_directions / atleast_kd(self.source_norms, self.ndim)

        self.stats_spherical_adversarial = ArrayQueue(maxlen=100, N=self.N)
        self.stats_step_adversarial = ArrayQueue(maxlen=30, N=self.N)
    
        # create queues to track various statistic used to derive the state
        # success rate, step size, relative location, progress in episode
        self.stats_is_adv = deque(maxlen=30)
        self.dist_derivative = deque(maxlen=30)
        self.improve_time_avg = deque(maxlen=30)
        self.moving_avg_step_dist = deque(maxlen=30)

        observation = []
        observation.append(np.float32(0.0))
        observation.append(np.float32(1.0))
        observation.append(slope)
        observation.append(self.improve_avg)
        observation.append(self.gain_moving)

        return observation

    def step(self, action):
        self.iter += 1
        # print(self.iter)
        # print(self.source_steps)

            
        # scale action
        action[0] = (action[0] + 2) / 4
        action[1] = (action[1] + 2) / 4
        
        if self.iter < 30:
            print(action)
        
        self.converged = self.dist < self.epsilon
        if self.converged or self.iter > self.steps:
            self.tb.close()
            self.done = True
            
        if self.nonadaptive:
            action[0], action[1] = 1, 1

        source = self.source_steps * action[0]        
        spherical = self.spherical_steps * action[1]
 
        # Draw new proposals
        self.candidates, self.spherical_candidates = draw_proposals(
            self.bounds,
            self.originals,
            self.best_advs,
            self.unnormalized_source_directions,
            self.source_directions,
            self.source_norms,
            spherical,
            source
            )
        
        # only check spherical candidates every k+1 steps
        self.check_spherical_and_update_stats = self.iter % (self.update_stats_every_k + 1) == 0
        # self.return_spherical = (self.iter - 1) % self.update_stats_every_k == 0

        if self.check_spherical_and_update_stats and self.nonadaptive:
            self.spherical_is_adv = self.is_adversarial(self.spherical_candidates.raw.unsqueeze(1))
            # print(self.spherical_is_adv)
            self.stats_spherical_adversarial.append(ep.astensor(self.spherical_is_adv))
            self.stats_step_adversarial.append(ep.astensor(self.is_adv))
            # update stats only in the vanilla version
            if self.nonadaptive:
                self.update_stats()
            obs = self.observation()
            r = self.reward(2)
            # gym step returns: observation, reward, done, info
            info = {"episode_number" : self.iter,
                    "epsilon" : self.dist,
                    "actions" : action,
                    "correct" : True,
                    "success" : self.success}
            return obs, r, self.done, info
        else:
            self.is_adv = self.is_adversarial(self.candidates.raw.unsqueeze(1))
            self.stats_is_adv.append(self.is_adv.numpy()[0])
            # self.stats_step_adversarial.append(self.is_adv)
        
        # in theory, we are closer per construction
        # but limited numerical precision might break this
        self.distances = ep.norms.l2(flatten(self.originals - self.candidates), axis=-1)
        self.closer = self.distances < self.source_norms
        # print(action, self.lastStep)
        # print(self.closer, self.is_adv)
        is_best_adv = self.is_adv and self.closer
        # print(is_best_adv)
            
        cond = not self.converged and is_best_adv
        # print(cond)
        if cond:
            self.gain = l2(self.candidates, self.best_advs).numpy()
            self.best_advs = self.candidates
            self.dist = l2(self.best_advs, self.wanted_point).numpy()
            self.gain_moving = self.gain_moving * 0.8 + (self.gain[0] * 0.2) / self.gap[0]
            self.improve_avg = self.improve_avg * 0.8 + (self.improve_last * 0.2) / self.steps
            self.improve_last = 0
        else:
            self.improve_last += 1
            self.gain = 0
            
        is_within_eps = self.dist < self.epsilon # check if perturbation < eps    
        # print(is_within_eps)
        if is_best_adv.numpy()[0] and is_within_eps:
            self.done = True
            self.success = True
            # print('success')
        
        self.unnormalized_source_directions = self.originals - self.best_advs
        self.source_norms = ep.norms.l2(flatten(self.unnormalized_source_directions), axis=-1)
        self.source_directions = self.unnormalized_source_directions / atleast_kd(self.source_norms, self.ndim)
        # update tensorboard
        self.update_tb(is_best_adv, cond)
        
        # if self.done:
        #     plt.imshow(self.best_advs[0].squeeze().numpy())
        #     plt.show(block=False)
        
        obs = self.observation()
        r = self.reward(2)
        # gym step returns: observation, reward, done, info
        # print(self.dist)
        # if self.iter % 100 == 0 or self.iter == 1:
        #     print(self.iter)
            # print(obs)
            # print(self.dist)
            # print(action[0], '|', action[1])
        info = {"episode_number" : self.iter,
                "epsilon" : self.dist,
                "actions" : action,
                "correct" : True,
                "success" : self.success}
        return obs, r, self.done, info
       
    def observation(self):
        # generate observation based on the history of responses
        
        # History of success/fail, goal and/or distance to goal, (history of step sizes)
        
        # Use dist in place of moving dist
        self.dist_moving = self.dist_moving * 0.8 + (self.dist / self.gap) * 0.2
        loc = self.dist / self.gap
        slope = self.dist_moving - loc
        

        observation = []
        observation.append(np.mean(self.stats_is_adv))
        observation.append(loc[0])
        observation.append(slope[0])
        observation.append(self.improve_avg)
        observation.append(self.gain_moving)
        # observation.appned(self.gain)
        # observation.append(self.iter / self.steps)
        # print(observation)
        return observation

    def reward2(self):
        fraction = self.dist / self.gap
        fraction_previous = (self.dist + self.gain) / self.gap
        reward = (1 - fraction ** 0.5) ** 2 - (1 - fraction_previous ** 0.5) ** 2
        return reward

    def reward3(self):
        fraction = self.dist / self.gap
        reward = (1 - fraction ** 0.5) ** 2
        return reward

    def reward4(self):
        reward = max([1 - self.iter / self.steps, 0.2]) * self.reward2()
        return reward

    def reward5(self):
        reward = 0
        if self.iter >= self.steps:
            reward = abs(math.log(self.dist / self.gap))
        return reward

    def reward(self, reward_nr):
        # R1
        if self.gain > 0:
            reward = 0.5 + self.gain / self.gap
        else:
            reward = 0

        if reward_nr == 2:
            # R2
            reward = self.reward2()
        elif reward_nr == 3:
            # R3
            reward = self.reward3()
        elif reward_nr == 4:
            # R4
            reward = self.reward4()
        elif reward_nr == 5:
            # R5
            reward = self.reward5()
        return reward
            
    def update_stats(self):
        self.tb.probability("spherical_is_adv", self.spherical_is_adv, self.iter)
        full = self.stats_spherical_adversarial.isfull()
        self.tb.probability("spherical_stats/full", full, self.iter)
        if full.any():
            probs = self.stats_spherical_adversarial.mean()
            # print(self.iter)
            # print(probs)
            cond1 = ep.logical_and(probs > 0.5, full)
            self.spherical_steps = ep.where(cond1, self.spherical_steps * self.step_adaptation, self.spherical_steps)
            self.source_steps = ep.where(cond1, self.source_steps * self.step_adaptation, self.source_steps)
            cond2 = ep.logical_and(probs < 0.2, full)
            self.spherical_steps = ep.where(cond2, self.spherical_steps / self.step_adaptation, self.spherical_steps)
            self.source_steps = ep.where(cond2, self.source_steps / self.step_adaptation, self.source_steps)
            self.stats_spherical_adversarial.clear(ep.logical_or(cond1, cond2))
            self.tb.conditional_mean("spherical_stats/isfull/success_rate/mean", probs, full, self.iter)
            self.tb.probability_ratio("spherical_stats/isfull/too_linear", cond1, full, self.iter)
            self.tb.probability_ratio("spherical_stats/isfull/too_nonlinear", cond2, full, self.iter)

        full = self.stats_step_adversarial.isfull()
        self.tb.probability("step_stats/full", full, self.iter)
        if full.any():
            probs = self.stats_step_adversarial.mean()
            # print(self.iter)
            # print(probs)
            # TODO: algorithm: changed the two values because we are currently tracking p(source_step_sucess)
            # instead of p(source_step_success | spherical_step_sucess) that was tracked before
            cond1 = ep.logical_and(probs > 0.25, full)
            self.source_steps = ep.where(cond1, self.source_steps * self.step_adaptation, self.source_steps)
            cond2 = ep.logical_and(probs < 0.1, full)
            self.source_steps = ep.where(cond2, self.source_steps / self.step_adaptation, self.source_steps)
            self.stats_step_adversarial.clear(ep.logical_or(cond1, cond2))
            self.tb.conditional_mean("step_stats/isfull/success_rate/mean", probs, full, self.iter)
            self.tb.probability_ratio("step_stats/isfull/success_rate_too_high", cond1, full, self.iter)
            self.tb.probability_ratio("step_stats/isfull/success_rate_too_low", cond2, full, self.iter)
                    
    def get_pair(self):
        startImgNr = random.randint(*self.indices)
        originImgNr = random.randint(*self.indices)
        
        # Make sure original image is correctly classified by the model
        while not ep.argmax(self.model(self.dataset[originImgNr][0].unsqueeze(1))).detach().numpy() == self.dataset[originImgNr][1]:
            originImgNr = random.randint(*self.indices)
        
        # Make sure starting and original images do not belong to the same class, and starting is correctly classified
        while self.dataset[startImgNr][1] == self.dataset[originImgNr][1] \
            or not ep.argmax(self.model(self.dataset[startImgNr][0].unsqueeze(1))).detach().numpy() == self.dataset[startImgNr][1]:
            startImgNr = random.randint(*self.indices)
        
        startImg = self.dataset[startImgNr][0].to(device)
        startLabel = self.dataset[startImgNr][1]
        
        originImg = self.dataset[originImgNr][0].to(device)
        originLabel = self.dataset[originImgNr][1]
        
        # startImg = self.dataset[1][0]
        # startLabel = self.dataset[1][1]
        # originImg = self.dataset[3][0]
        # originLabel = self.dataset[3][1]
        
        return startImg, startLabel, originImg, originLabel
    
    def update_tb(self, is_best_adv, cond):
        self.tb.probability("converged", self.converged, self.iter)
        self.tb.scalar("updated_stats", self.check_spherical_and_update_stats, self.iter)
        self.tb.histogram("norms", self.source_norms, self.iter)
        self.tb.probability("is_adv", self.is_adv, self.iter)
        self.tb.histogram("candidates/distances", self.distances, self.iter)
        self.tb.probability("candidates/closer", self.closer, self.iter)
        self.tb.probability("candidates/is_best_adv", is_best_adv, self.iter)
        self.tb.probability("new_best_adv_including_converged", is_best_adv, self.iter)
        self.tb.probability("new_best_adv", cond, self.iter)

        self.tb.histogram("spherical_step", self.spherical_steps, self.iter)
        self.tb.histogram("source_step", self.source_steps, self.iter)
    


# for env in gym.envs.registry.env_specs:
#     if 'BoundaryStep-v0' not in env:
#         register(
#             id='BoundaryStep-v0',
#             entry_point='boundarystep.envs:BoundaryStep',
#             reward_threshold=0.95
#             )