import eagerpy as ep
import numpy as np
import gym
import torch
import random
from torchvision import datasets, transforms
from typing import Union, Tuple, Optional, Any
from foolbox import PyTorchModel
from foolbox.tensorboard import TensorBoard
from foolbox.attacks import BoundaryAttack
from foolbox.attacks.boundary_attack import ArrayQueue, draw_proposals
from foolbox.attacks.base import get_is_adversarial
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.registration import register
from utils.buckets import Buckets
from foolbox.criteria import TargetedMisclassification
from utils.utils import MisdirectedMisclassification, iskl_adversarial
from utils.utils import flatten, atleast_kd
from models.trainMNISTtorch import Net
from joblib import load
import math

class BoundaryStep(gym.Env):
    def __init__(self, logdir='/logs'):
        super(BoundaryStep, self).__init__()
        self.buckets = Buckets(nrBuckets=10)
        
        # Actions controlled by the interceptor
        self.action_space = spaces.Discrete(3)
        # Observation space is the MNIST inputs
        self.observation_space = spaces.Box(low=0, high=10, shape=(465,), dtype=np.float32)

        # Load MNIST pytorch CNN model -- 99.1% acc
        transform=transforms.ToTensor()
        self.dataset = datasets.MNIST('../data', train=False, transform=transform, download=True)
        self.mode = Net()
        self.mode.load_state_dict(torch.load('models/mnist_cnn.pt'))
        self.mode.eval()
        self.model = PyTorchModel(self.mode, bounds=(0, 1))
        
        # Load MNIST sklearn RF model -- 97.1% acc
        self.sub = load('models/RF.joblib')
        
        self.actions = [0, 0, 0]
        self.done = False
        self.logdir = logdir
        
    def reset(self, early_stop: Optional[float] = None):
        self.attack = BoundaryAttack()
        self.tb = TensorBoard(logdir=self.logdir)
        self.early_stop = early_stop
        self.starting_point, self.wanted_point, startLabel = self.get_pair()
        self.starting_point, _ = ep.astensor_(self.starting_point)
        print("Start label:", startLabel)
        # check for correcteness in how to supply the startLabel
        self.criterion = TargetedMisclassification(torch.tensor([startLabel]))
        self.misterion = MisdirectedMisclassification(torch.tensor([startLabel]))
        self.sklerion = iskl_adversarial([startLabel], self.sub)
        self.originals, self.restore_type = ep.astensor_(self.wanted_point)
        self.iter = 0
        self.done = False

        self.is_adversarial = get_is_adversarial(self.criterion, self.model)
        self.mis_adversarial = get_is_adversarial(self.misterion, self.model)
        

        if self.starting_point is None:
            raise ValueError("no starting_point provided")
        else:
            self.best_advs = ep.astensor(self.starting_point)
            del self.starting_point

        is_adv = self.is_adversarial(self.best_advs.raw.unsqueeze(1))
        if not is_adv:
            raise ValueError("starting_point is not adversarial")

        self.N = len(self.originals)
        self.ndim = self.originals.ndim
        self.bounds = self.model.bounds
        self.spherical_steps = ep.ones(self.originals, self.N) * self.attack.spherical_step
        self.source_steps = ep.ones(self.originals, self.N) * self.attack.source_step
        self.unnormalized_source_directions = self.originals - self.best_advs
        self.source_norms = ep.norms.l2(flatten(self.unnormalized_source_directions), axis=-1)
        self.source_directions = self.unnormalized_source_directions / atleast_kd(self.source_norms, self.ndim)
    
        # Draw first batch of candidates so the first step has something to act on
        self.candidates, self.spherical_candidates = draw_proposals(
            self.bounds,
            self.originals,
            self.best_advs,
            self.unnormalized_source_directions,
            self.source_directions,
            self.source_norms,
            self.spherical_steps,
            self.source_steps,
            )

        # tb.scalar("batchsize", N, 0)

        # create two queues for each sample to track success rates
        # (used to update the hyper parameters)
        self.stats_spherical_adversarial = ArrayQueue(maxlen=100, N=self.N)
        self.stats_step_adversarial = ArrayQueue(maxlen=30, N=self.N)

    def step(self, actionID):
        #TODO: throw some benign related queries and evaluate over different ratios of collisions
        self.iter += 1
        self.converged = self.source_steps < self.attack.source_step_convergance
        if self.converged or self.iter > self.attack.steps:
            self.tb.close()
            return True, self.best_advs  # pragma: no cover
        self.converged = atleast_kd(self.converged, self.ndim)

        self.unnormalized_source_directions = self.originals - self.best_advs
        self.source_norms = ep.norms.l2(flatten(self.unnormalized_source_directions), axis=-1)
        self.source_directions = self.unnormalized_source_directions / atleast_kd(self.source_norms, self.ndim)

        # only check spherical candidates every k steps
        self.check_spherical_and_update_stats = self.iter % self.attack.update_stats_every_k == 0
        self.return_spherical = (self.iter - 1) % self.attack.update_stats_every_k == 0

        if not self.check_spherical_and_update_stats:
            # Order of query and response:
            # Check is_adv with prev candidate, then find new candidates
            self.is_adv = self.switch(actionID, self.candidates.raw.unsqueeze(1))
            self.candidates, self.spherical_candidates = draw_proposals(
                self.bounds,
                self.originals,
                self.best_advs,
                self.unnormalized_source_directions,
                self.source_directions,
                self.source_norms,
                self.spherical_steps,
                self.source_steps,
                )
        else:
            assert self.spherical_candidates is not None
            self.spherical_is_adv = self.switch(actionID, self.spherical_candidates.raw.unsqueeze(1))
            print(self.spherical_is_adv.numpy().shape)
            print(self.N)
            self.stats_spherical_adversarial.append(self.spherical_is_adv)
            # TODO: algorithm: the original implementation ignores those samples
            # for which spherical is not adversarial and continues with the
            # next iteration -> we estimate different probabilities (conditional vs. unconditional)
            # TODO: thoughts: should we always track this because we compute it anyway
            # TODO: maybe move this to the main iteration in order to track the step better
            self.stats_step_adversarial.append(self.is_adv)
            self.update_stats()
            # next call is going to be source, so we return candidates
            obs, bIndex = self.observation(actionID, self.is_adv, self.candidates)
            r = self.reward(self.buckets.getAverageStepSizeBucket(bIndex), self.iter)
            # gym step returns: observation, reward, done, info
            return obs, r, False, {"episode":self.iter}
        
        # in theory, we are closer per construction
        # but limited numerical precision might break this
        self.distances = ep.norms.l2(flatten(self.originals - self.candidates), axis=-1)
        self.closer = self.distances < self.source_norms
        print(self.closer, self.is_adv)
        is_best_adv = ep.logical_and(self.is_adv, self.closer)
        is_best_adv = atleast_kd(is_best_adv, self.ndim)
        
        print(is_best_adv)
        cond = self.converged.logical_not().logical_and(is_best_adv)
        self.best_advs = ep.where(cond, self.candidates, self.best_advs)

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
        
        obs, bIndex = self.observation(actionID, self.is_adv, self.candidates)
        r = self.reward(self.buckets.getAverageStepSizeBucket(bIndex), self.iter)
        # gym step returns: observation, reward, done, info
        return obs, r, False, {"episode":self.iter}
    
    def observation(self, actionID, is_adv, candidate):
        # return state based on the next candidate generated by the boundary attack
        x, restore_type = ep.astensor_(candidate)
        print(x.raw.unsqueeze(1).shape)
        is_misdirection = self.check_misdirection(actionID, is_adv)
        bucketIndex = self.buckets.addQuery(x.raw.unsqueeze(3), actionID, is_misdirection)
        state = self.buckets.getState(bucketIndex)

        return state, bucketIndex
    
    def reward(self, averageStepsize, queryCounter):
        # reward is based on average stepsize of adversarial queries
        if averageStepsize <= 0:
            print("too low average stepsize: ", averageStepsize)
            averageStepsize = 0

        if averageStepsize >= 1:
            r = 0#-queryCounter/1000
        else:
            #print(averageStepsize)
            r = abs(math.log(averageStepsize,10))#-queryCounter/1000
            
        # penalize benign queries being misclassified
        # if returnedLabel != realLabel:
        #     return -5
        # else:
        #     return 0
        
        return r

    def switch(self, actionID, candidates):
        print('Action:', actionID)
        if self.iter < 30:
            actionID = 0
        if actionID == 0:
            is_adv = self.is_adversarial(candidates)
            self.actions[0] += 1
        if actionID == 1:
            is_adv = self.sklerion(candidates)
            self.actions[1] += 1
        if actionID == 2:
            is_adv = self.mis_adversarial(candidates)
            self.actions[2] += 1
        print('Is adversarial:', is_adv)
        return ep.astensor(torch.tensor(is_adv))
    
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
        startImgNr = random.randint(0,10000)
        originImgNr = random.randint(0,10000)
        
        # Make sure original image is correctly classified by the model
        while not ep.argmax(self.model(self.dataset[originImgNr][0].unsqueeze(1))).detach().numpy() == self.dataset[originImgNr][1]:
            originImgNr = random.randint(0,1000)
        
        # Make sure starting and original images do not belong to the same class, and starting is correctly classified
        while self.dataset[startImgNr][1] == self.dataset[originImgNr][1] \
            or not ep.argmax(self.model(self.dataset[startImgNr][0].unsqueeze(1))).detach().numpy() == self.dataset[startImgNr][1]:
            startImgNr = random.randint(0,1000)
        
        startImg = self.dataset[startImgNr][0]
        startLabel = self.dataset[startImgNr][1]
        
        originImg = self.dataset[originImgNr][0]
        
        return startImg, originImg, startLabel
        

# for env in gym.envs.registry.env_specs:
#     if 'BoundaryStep-v0' not in env:
#         register(
#             id='BoundaryStep-v0',
#             entry_point='boundarystep.envs:BoundaryStep',
#             reward_threshold=0.95
#             )