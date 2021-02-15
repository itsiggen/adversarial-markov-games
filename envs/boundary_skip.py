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

random.seed(2)

class BoundarySkip(gym.Env):
    def __init__(
        self,
        steps: int = 5000,
        spherical_step: float = 2e-2,
        source_step: float = 2e-2,
        tensorboard = False,
        update_stats_every_k: int = 10
        ):
        super(BoundarySkip, self).__init__()
        
        # Boundary Attack inits
        self.steps = steps
        self.spherical_step = spherical_step
        self.source_step = source_step
        self.tensorboard = tensorboard
        self.update_stats_every_k = update_stats_every_k
        
        # Actions controlled by the adversary
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        # Observation space is the MNIST inputs
        self.observation_space = spaces.Box(low=0, high=1, shape=(30,), dtype=np.float32)

        # Load MNIST pytorch CNN model -- 99.1% acc
        transform=transforms.ToTensor()
        self.dataset = datasets.MNIST('../data', train=False, transform=transform, download=True)
        self.mode = Net()
        self.mode.load_state_dict(torch.load('models/mnist_cnn.pt'))
        self.mode.eval()
        self.model = PyTorchModel(self.mode, bounds=(0, 1))
    
        self.done = False
        
    def reset(self):
        self.tb = TensorBoard(logdir=self.tensorboard)
        self.epsilon = 2
        self.starting_point, startLabel, self.wanted_point, originLabel = self.get_pair()
        self.starting_point, _ = ep.astensor_(self.starting_point)
        print("Start label:", startLabel)
        print("Wanted label:", originLabel)
        # check for correcteness in how to supply the startLabel
        self.criterion = TargetedMisclassification(torch.tensor([startLabel]))
        self.originals, self.restore_type = ep.astensor_(self.wanted_point)
        self.init_dist = l2(self.starting_point, self.wanted_point)
        self.iter = 0
        self.done = False

        self.is_adversarial = get_is_adversarial(self.criterion, self.model)
        
        if self.starting_point is None:
            raise ValueError("no starting_point provided")
        else:
            self.best_advs = ep.astensor(self.starting_point)

        is_adv = self.is_adversarial(self.best_advs.raw.unsqueeze(1))
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
    
        # create queues to track various statistic used to derive the state
        # success rate, step size, relative location, progress in episode
        self.stats_spherical_adversarial = deque(maxlen=100)
        self.stats_step_adversarial = deque(maxlen=30)
        self.best_adv_dist = deque(maxlen=30)
        self.dist_derivative = deque(maxlen=30)
        self.improve_time_avg = deque(maxlen=30)
        self.moving_avg_step_dist = deque(maxlen=30)

    def step(self, action):
        self.iter += 1
        # print(self.iter)
        # print(self.source_steps)
        if self.iter > self.steps:
            self.tb.close()
            self.done = True
        
        # Draw new proposals
        self.candidates, self.spherical_candidates = draw_proposals(
            self.bounds,
            self.originals,
            self.best_advs,
            self.unnormalized_source_directions,
            self.source_directions,
            self.source_norms,
            self.spherical_steps*action[1],
            self.source_steps*action[0],
            )
        
        # only check spherical candidates every k+1 steps
        self.check_spherical_and_update_stats = self.iter % (self.update_stats_every_k + 1) == 0
        # self.return_spherical = (self.iter - 1) % self.update_stats_every_k == 0

        if self.check_spherical_and_update_stats:
            assert self.spherical_candidates is not None
            self.spherical_is_adv = self.is_adversarial(self.spherical_candidates.raw.unsqueeze(1))
            # print(self.spherical_is_adv)
            self.stats_spherical_adversarial.append(self.spherical_is_adv)
            # update stats only in the vanilla version
            # self.update_stats()
            obs = self.observation(self.spherical_is_adv)
            r = self.reward()
            # gym step returns: observation, reward, done, info
            return obs, r, self.done, {"episode":self.iter}
        else:
            self.spherical_is_adv = None
            self.is_adv = self.is_adversarial(self.candidates.raw.unsqueeze(1))
            self.stats_step_adversarial.append(self.is_adv)
        
        # in theory, we are closer per construction
        # but limited numerical precision might break this
        self.distances = ep.norms.l2(flatten(self.originals - self.candidates), axis=-1)
        self.closer = self.distances < self.source_norms
        # print(self.closer, self.is_adv)
        is_best_adv = ep.logical_and(self.is_adv, self.closer)
        is_best_adv = atleast_kd(is_best_adv, self.ndim)
        print(is_best_adv)
            
        cond = self.converged.logical_not().logical_and(is_best_adv)
        self.best_advs = ep.where(cond, self.candidates, self.best_advs)

        # check if perturbation < eps
        dist = l2(self.best_advs, self.wanted_point)
        # dista = ep.norms.linf(flatten(self.best_advs - self.wanted_point), axis=-1)
        is_within_eps = dist < self.epsilon
        if self.iter % 109 == 0:
            # print(is_within_eps.numpy()[0])
            print(self.iter)
            print(dist)
        # print(dista)
        # print(is_within_eps)
        self.done = is_best_adv.numpy()[0] and is_within_eps.numpy()[0]
        
        self.unnormalized_source_directions = self.originals - self.best_advs
        self.source_norms = ep.norms.l2(flatten(self.unnormalized_source_directions), axis=-1)
        self.source_directions = self.unnormalized_source_directions / atleast_kd(self.source_norms, self.ndim)

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
        
        # if self.done:
        #     plt.imshow(self.best_advs[0].squeeze().numpy())
        #     plt.show(block=False)
        
        obs = self.observation()
        r = self.reward()
        # gym step returns: observation, reward, done, info
        return obs, r, self.done, {"episode_number":self.iter}
       
    def observation(self):
        # generate observation based on the history of responses
        
        # History of success/fail, goal and/or distance to goal, (history of step sizes)

        self.dist_between_step_slope = self.dist_between_step_slope * 0.8 + (
                self.dist_opt_adv / self.dist_init_adv) * 0.2
        dist_derivative = self.dist_between_step_slope - self.dist_opt_adv / self.dist_init_adv

        if self.dist_between_step > 0:
            self.dist_between_step_moving_avg = self.dist_between_step_moving_avg * 0.8 + (
                    self.dist_between_step * 0.2) / self.dist_init_adv
            self.improve_time_avg = self.improve_time_avg * 0.8 + (self.improve_time_last * 0.2) / self.query_max
            self.improve_time_last = 0
            # ============== OPTIONAL ==================
            # self.state_mask = np.append(self.action_mask_factor, self.state_mask[:-1])
            # self.state_mask_success = np.append(self.action_mask_factor,
            #                                             self.state_mask_success[:-1])
        else:
            self.improve_time_last += 1
            # ============== OPTIONAL ==================
            # self.state_mask = np.append(-self.action_mask_factor, self.state_mask[:-1])

        observation = []

        # ============== FIXED ==================
        observation = np.append(observation, self.dist_opt_adv / self.dist_init_adv)
        observation = np.append(observation, self.query_current / self.query_max)
        observation = np.append(observation, dist_derivative)
        observation = np.append(observation, self.improve_time_avg)
        observation = np.append(observation, self.dist_between_step_moving_avg)
        return observation

    
    def reward(self, averageStepsize, queryCounter):
        # reward is based on finding best_advs and the l2 of the improvement

        r = 0#-queryCounter/1000
        
        return r
    
    def check_misdirection(self, actionID, is_adv):
        # method to return if misdirection occured
        # either actively or by the substitute model
        if actionID == 2:
            return 1
        elif actionID == 1 and not is_adv:
            return 1
        else:
            return 0
    
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
        # startImgNr = random.randint(0,10000)
        # originImgNr = random.randint(0,10000)
        
        # # Make sure original image is correctly classified by the model
        # while not ep.argmax(self.model(self.dataset[originImgNr][0].unsqueeze(1))).detach().numpy() == self.dataset[originImgNr][1]:
        #     originImgNr = random.randint(0,1000)
        
        # # Make sure starting and original images do not belong to the same class, and starting is correctly classified
        # while self.dataset[startImgNr][1] == self.dataset[originImgNr][1] \
        #     or not ep.argmax(self.model(self.dataset[startImgNr][0].unsqueeze(1))).detach().numpy() == self.dataset[startImgNr][1]:
        #     startImgNr = random.randint(0,1000)
        
        # startImg = self.dataset[startImgNr][0]
        # startLabel = self.dataset[startImgNr][1]
        
        # originImg = self.dataset[originImgNr][0]
        # originLabel = self.dataset[originImgNr][1]
        
        startImg = self.dataset[1][0]
        startLabel = self.dataset[1][1]
        originImg = self.dataset[3][0]
        originLabel = self.dataset[3][1]
        
        return startImg, startLabel, originImg, originLabel
        

# for env in gym.envs.registry.env_specs:
#     if 'BoundaryStep-v0' not in env:
#         register(
#             id='BoundaryStep-v0',
#             entry_point='boundarystep.envs:BoundaryStep',
#             reward_threshold=0.95
#             )