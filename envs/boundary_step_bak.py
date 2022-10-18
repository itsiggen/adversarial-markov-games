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
from foolbox.distances import l2
from gym import error, spaces, utils
from utils.buckets import Buckets
from foolbox.criteria import TargetedMisclassification
from utils.utils import flatten, atleast_kd
from utils.utils import get_is_adversarial
from models.trainMNISTtorch import Net
from joblib import load
import matplotlib.pyplot as plt
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BoundaryStep(gym.Env):
    def __init__(
        self,
        steps: int = 5000,
        spherical_step: float = 1e-2,
        source_step: float = 1e-2,
        source_step_convergence: float = 1e-7,
        step_adaptation: float = 1.5,
        ratio_benign = 0,
        train = True,
        tensorboard = False,
        update_stats_every_k: int = 10
        ):
        super(BoundaryStep, self).__init__()
        
        # Boundary Attack inits
        self.steps = steps
        self.spherical_step = spherical_step
        self.source_step = source_step
        self.source_step_convergence = source_step_convergence
        self.step_adaptation = step_adaptation
        self.ratio_benign = ratio_benign
        self.tensorboard = tensorboard
        self.update_stats_every_k = update_stats_every_k
        
        # Actions controlled by the interceptor
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        # Observation space is the MNIST inputs
        # self.observation_space = spaces.Box(low=0, high=10, shape=(465,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        # Load MNIST pytorch CNN model -- 99.1% acc
        transform=transforms.ToTensor()
        self.dataset = datasets.MNIST('../data', train=False, transform=transform, download=True)
        self.mode = Net()
        self.mode.load_state_dict(torch.load('models/mnist_cnn.pt'))
        self.mode.eval()
        # for param in self.mode.parameters():
        #     param.requires_grad = False
        self.model = PyTorchModel(self.mode, bounds=(0, 1))
        self.indices = [0,7999] if train else [8000,9999]
        
        self.done = False
        
    def reset(self):
        # self.attack = BoundaryAttack()
        self.tb = TensorBoard(logdir=self.tensorboard)
        self.epsilon = 2
        self.starting_point, startLabel, self.wanted_point, originLabel = self.get_pair()
        self.starting_point, _ = ep.astensor_(self.starting_point)
        print("Start label:", startLabel)
        print("Wanted label:", originLabel)
        # check for correcteness in how to supply the startLabel
        self.criterion = TargetedMisclassification(torch.tensor([startLabel]))
        self.originals, self.restore_type = ep.astensor_(self.wanted_point)
        self.gap = l2(self.starting_point, self.wanted_point).numpy()
        self.buckets = Buckets(nrBuckets=10)
        self.iter = 0
        self.action = 0
        self.done = False
        self.success = False
        self.r = []
        self.avgstep = []
        self.diff = []

        self.is_adversarial = get_is_adversarial(self.criterion, self.model)
        

        if self.starting_point is None:
            raise ValueError("no starting_point provided")
        else:
            self.best_advs = ep.astensor(self.starting_point)

        is_adv, self.logits = self.is_adversarial(self.best_advs.raw.unsqueeze(1))
        if not is_adv:
            raise ValueError("starting_point is not adversarial")
        
        self.check_candidates = is_adv
        self.N = len(self.originals) # must be 1 as we perform 1 attack at a time
        self.ndim = self.originals.ndim
        self.bounds = self.model.bounds
        self.spherical_steps = ep.ones(self.originals, self.N) * self.spherical_step
        self.source_steps = ep.ones(self.originals, self.N) * self.source_step
        self.unnormalized_source_directions = self.originals - self.best_advs
        self.source_norms = ep.norms.l2(flatten(self.unnormalized_source_directions), axis=-1)
        self.source_directions = self.unnormalized_source_directions / atleast_kd(self.source_norms, self.ndim)
    
        # # Draw first batch of candidates so the first step has something to act on
        # self.candidates, self.spherical_candidates = draw_proposals(
        #     self.bounds,
        #     self.originals,
        #     self.best_advs,
        #     self.unnormalized_source_directions,
        #     self.source_directions,
        #     self.source_norms,
        #     self.spherical_steps,
        #     self.source_steps,
        #     )

        # tb.scalar("batchsize", N, 0)

        # create two queues for each sample to track success rates
        # (used to update the hyper parameters)
        self.stats_spherical_adversarial = ArrayQueue(maxlen=100, N=self.N)
        self.stats_step_adversarial = ArrayQueue(maxlen=30, N=self.N)
        
        # obs, bIndex = self.observation(2, True, self.candidates)
        # obs = np.zeros(10)
        self.candidates = self.best_advs
        obs, bIndex, self.lastStep = self.observation(0, self.logits, is_adv, self.best_advs)
        return obs
    
    def step(self, action):
        if random.random() < self.ratio_benign:
            # throw a benign query with P = ratio_benign
        else:
            self.iter += 1
            # print(self.iter)
            self.converged = self.source_steps < self.source_step_convergence
            self.converged = atleast_kd(self.converged, self.ndim)
            # print(self.source_steps)
            if self.converged or self.iter > self.steps:
                self.tb.close()
                self.done = True
    
            # only check spherical candidates every k+1 steps
            self.check_spherical_and_update_stats = self.iter % (self.update_stats_every_k + 1)
            
            if self.check_spherical_and_update_stats == 1 and self.iter != 1:
                # sto 12 apofasizei gia to spherical tou 11
                self.spherical_is_adv = self.switch(self.check_spherical, self.spherical_candidates, self.lastStep, action)
                # print(self.spherical_is_adv)
                self.stats_spherical_adversarial.append(self.spherical_is_adv)
                # TODO: algorithm: the original implementation ignores those samples
                # for which spherical is not adversarial and continues with the
                # next iteration -> we estimate different probabilities (conditional vs. unconditional)
                # TODO: thoughts: should we always track this because we compute it anyway
                # TODO: maybe move this to the main iteration in order to track the step better
                # self.stats_step_adversarial.append(self.is_adv)
                self.update_stats()
            else:
                # sto 11 apofasizei gia to candidate tou 10, sto 10 gia tou 9..
                self.is_adv = self.switch(self.check_candidates, self.candidates, self.lastStep, action)
                self.stats_step_adversarial.append(self.is_adv)
    
            if self.check_spherical_and_update_stats == 0:
                # sto 11 dialegei kai epistrefei spherical
                assert self.spherical_candidates is not None
                self.check_spherical, self.logits = self.is_adversarial(self.spherical_candidates.raw.unsqueeze(1))
                # obs = torch.nn.functional.softmax(self.logits, dim=1)
                obs, bIndex, self.lastStep = self.observation(self.action, self.logits, self.check_spherical, self.spherical_candidates)
                r = self.reward(self.buckets.getStepSizeBucket(bIndex), self.iter)
                # gym step returns: observation, reward, done, info
                info = {"iteration" : self.iter,
                        "epsilon" : self.dist,
                        "actions" : self.action,
                        "success" : self.success}
                return obs, r, self.done, info
            
            # in theory, we are closer per construction
            # but limited numerical precision might break this
            self.distances = ep.norms.l2(flatten(self.originals - self.candidates), axis=-1)
            self.closer = self.distances < self.source_norms
            # print(self.closer, self.is_adv)
            is_best_adv = ep.logical_and(self.is_adv, self.closer)
            is_best_adv = atleast_kd(is_best_adv, self.ndim)
            # print(is_best_adv)
                
            cond = self.converged.logical_not().logical_and(is_best_adv)
            self.best_advs = ep.where(cond, self.candidates, self.best_advs)
    
            # check if perturbation < eps
            self.dist = l2(self.best_advs, self.wanted_point)
            # dista = ep.norms.linf(flatten(self.best_advs - self.wanted_point), axis=-1)
            is_within_eps = self.dist < self.epsilon
            if self.iter % 100 == 0 or self.iter == 1:
                # print(is_within_eps.numpy()[0])
                print(self.iter)
                print(self.dist.numpy())
                print(torch.nn.functional.softmax(self.logits, dim=1))
            # print(dista)
            # print(is_within_eps)
            if is_best_adv.numpy()[0] and is_within_eps.numpy()[0]:
                self.done = True
                self.success = True
                # print('success')
            
            self.unnormalized_source_directions = self.originals - self.best_advs
            self.source_norms = ep.norms.l2(flatten(self.unnormalized_source_directions), axis=-1)
            self.source_directions = self.unnormalized_source_directions / atleast_kd(self.source_norms, self.ndim)
            
            # Draw new proposals
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
            
            # Check if candidate is adversarial    
            self.check_candidates, self.logits = self.is_adversarial(self.candidates.raw.unsqueeze(1))
    
            obs, bIndex, self.lastStep = self.observation(self.action, self.logits, self.check_candidates, self.candidates)
            r = self.reward(self.buckets.getStepSizeBucket(bIndex), self.iter)
            # gym step returns: observation, reward, done, info
            info = {"episode_number" : self.iter,
                        "epsilon" : self.dist.numpy(),
                        "actions" : self.action,
                        "success" : self.success}
            return obs, r, self.done, info
    
    def observation(self, action, logits, is_adv, candidate):
        # return state based on the next candidate generated by the boundary attack
        # if self.iter % 5 == 0:
        #     candidate = self.throw_benign()
        x, restore_type = ep.astensor_(candidate)
        logs = torch.nn.functional.softmax(logits, dim=1)
        bucketIndex = self.buckets.addQuery(x.raw.unsqueeze(3), action, logs, False)
        lastStep = self.buckets.getLastStepBucket(bucketIndex)

        return logs, bucketIndex, lastStep
    
    def reward(self, stepsize, queryCounter):
        # Reward possibilities:
        # Closer to the initial example, higher the smaller the stepsize diff is
        averageStepsize = np.mean(np.asarray(stepsize))
        
        if averageStepsize <= 0:
            print("too low average stepsize: ", averageStepsize)
            averageStepsize = 0

        # if self.iter <= 30:
        #     self.step_ref = averageStepsize*1.01
        
        if self.iter == 1:
            diff = [1]
        else:
            diff = [x - stepsize[-1] for x in stepsize]

        if averageStepsize >= 1:
            r = 0#-queryCounter/1000
        else:
            #print(averageStepsize)
            # r = abs(math.log(averageStepsize,10))#-queryCounter/1000
            # r = 10*averageStepsize
            # r = abs(math.log(abs(averageStepsize/self.step_ref - 1)),10)
            # r = abs(math.log(np.linalg.norm(diff), 10))
            # if self.done:
            #     r = l2(self.best_advs, self.wanted_point).numpy()*0.2
            #     print(r)
            #     print("DOOONE")
            # else:
            #     r = 0
            r = abs(math.log(self.gap - l2(self.starting_point, self.best_advs).numpy()) / self.gap) * 0.2
                
            
                        
        # penalize benign queries being misclassified
        # if returnedLabel != realLabel:
        #     return -5
        # else:
        #     return 0
        
        self.r.append(r)
        self.avgstep.append(averageStepsize)
        self.diff.append(np.linalg.norm(diff))
        
        # if self.iter == 1001:
        #     plt.plot(self.r, color='olive', label="rew")
        #     # plt.plot(self.avgstep, color='blue', label="step")
        #     # plt.plot(self.diff, color='red', label="diff")
        #     plt.show()
        
        # if self.iter % 100 == 0 or self.iter <= 100:
        #     # print(self.iter)
        #     print("stepsize", averageStepsize)
        #     print("diff", np.linalg.norm(diff))
        #     print("reward", r)            
        return r

    def switch(self, is_adv, candidate, step, action):
        # If is_adv and < hypersphere radius, then it's not
        # print(is_adv)
        if is_adv:        
            a = torch.tensor(action)
            b = torch.tensor(step)
            # print(a)
            # print(b)
            c = ep.astensor(a < b)
            # print(c)
            # print(is_adv.shape)
            # print(a.shape)
            return c
        else:
            return ep.astensor(is_adv)
    
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
    
    def throw_benign(self):
        nr=self.iter//100 + np.random.randint(0,100)
        
        mu, sigma = 0, 0.1 # mean and standard deviation
        s = np.random.normal(mu, sigma, 28*28)
        s = s.reshape(28,28,1)
        s = np.absolute(s)
        benign = np.add(self.dataset[nr][0], s)
        return benign
        

# for env in gym.envs.registry.env_specs:
#     if 'BoundaryStep-v0' not in env:
#         register(
#             id='BoundaryStep-v0',
#             entry_point='boundarystep.envs:BoundaryStep',
#             reward_threshold=0.95
#             )