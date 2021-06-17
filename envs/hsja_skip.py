import eagerpy as ep
import numpy as np
import gym
import torch
import random
from foolbox import PyTorchModel
from foolbox.tensorboard import TensorBoard
from foolbox.attacks.base import get_is_adversarial
from gym import spaces
from foolbox.criteria import TargetedMisclassification
from utils.utils import flatten, atleast_kd
from utils.queues import l2
import utils.perlin as pn
from typing import Union, Callable, List
from models.trainMNISTtorch import Net
from models.trainAdvMNISTtorch import LeNet5
from collections import deque
import matplotlib.pyplot as plt
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HsjaSkip(gym.Env):
    def __init__(
        self,
        steps: int = 1000,
        reps: int = 64,
        init_gradient_eval_steps: int = 100,
        max_gradient_eval_steps: int = 10000,
        gamma: float = 1.0,
        defended = False,
        nonadaptive = False,
        epsilon = 0.01,
        train = True,
        rewarder = 1,
        scale = 400,
        perlin = 0,
        dataset = None,
        tensorboard = False,
        update_stats_every_k: int = 10
        ):
        super(HsjaSkip, self).__init__()  

        # Boundary Attack inits
        self.steps = steps
        self.reps = reps
        self.init_grad_evals = init_gradient_eval_steps
        self.max_grad_evals = max_gradient_eval_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.nonadaptive = nonadaptive
        self.rewarder = rewarder
        self.scale = scale
        self.perlin = perlin
        self.tensorboard = tensorboard

        # Actions space
        self.action_space = spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float32)
        # Observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

        # Load MNIST pytorch CNN model -- 99.1% acc -- 98.9% acc adversarially trained
        self.dataset = dataset
        if defended:
            self.mode = LeNet5()
            self.mode.load_state_dict(torch.load('./models/mnist_cnn_adv.pt', map_location=torch.device('cpu')))
            self.mode.eval()
        else:
            self.mode = Net()
            self.mode.load_state_dict(torch.load('./models/mnist_cnn.pt'))
            self.mode.eval()

        self.model = PyTorchModel(self.mode, bounds=(0, 1))
        self.indices = [0,7999] if train else [8000,9999]
        self.dim = 28
        self.resets = 0
    
        self.done = False
    
    def reset(self):
        # Initialize new targeted attack 
        self.iter = 0
        self.reps = 0
        self.tb = TensorBoard(logdir=self.tensorboard)
        self.starting_point, startLabel, self.wanted_point, originLabel = self.get_pair()
        while np.isnan(self.starting_point).any() or np.isnan(self.wanted_point).any():
            self.starting_point, startLabel, self.wanted_point, originLabel = self.get_pair()
        # self.starting_point, _ = ep.astensor_(self.starting_point)
        # self.original, self.restore_type = ep.astensor_(self.wanted_point)
        # if self.resets < 3: print("Start:", startLabel, "| Wanted:", originLabel)
        self.criterion = TargetedMisclassification(torch.tensor([startLabel]))
        # Distance between starting and origin point / current best adv
        self.gap = l2(self.starting_point, self.wanted_point)
        self.dist = self.gap
        self.goal = self.gap * self.epsilon
        # Distance between successive steps
        self.diff = np.float32(0.0)
        # Moving average of the closing distance
        self.dist_moving = np.float32(1.0)
        # Initial mask
        self.x_mask = np.ones(self.wanted_point.shape, dtype=np.float32)
        # Moving average of the gain
        self.gain_moving = 0.1
        # Target epsilon as the ratio of initial l2 distance
        self.actions = []
        self.done = False
        self.success = False
        self.resets += 1

        self.is_adversarial = get_is_adversarial(self.criterion, self.model)
        
        if self.starting_point is None:
            raise ValueError("no starting_point provided")
        else:
            self.best_advs = self.starting_point
        
        is_adv = self.is_adversarial(self.best_advs.unsqueeze(1))
        if not is_adv:
            raise ValueError("starting_point is not adversarial")
          
        self.best_advs, self.extra_queries = self.binary_step(ep.astensor(self.best_advs), 1)    
        # pos = self.best_advs.squeeze(0).numpy()
        # pos = pos[::4,::4].flatten().tolist()
        
        observation = []
        # observation.append(np.float32(1.0))
        # TODO: encode source n target class in obs
        observation.append(np.float32(0.5))
        observation.append(np.float32(1.0))
        observation.append(np.float32(1.0))
        observation.append(np.float32(0.5))
        observation.append(np.float32(0.0))
        # observation.extend(pos)

        return observation
        
    def scale_binary(self, v):
        # Binary search parameter from [-2,2] to [0.01, 0.41]
        return (v + 2) / 10 + 0.01
    
    def scale_perlin(self, v):
        act = ((v + 2) / 4) * (self.dim - 2) + 1
        return np.nan_to_num(act, nan=0.0, posinf=self.dim-1, neginf=0.0)
    
    def scale_delta(self, v):
        # Delta from [-2,2] to [0.0001,0.0101]
        return ((v + 2) / self.scale) + 0.0001
    
    def scale_step(self, v):
        # Jump step search from [-2,2] to [0.1,0.9]
        return (v + 2) / 5 + 0.1
    
    # try different num of grad
    def scale_grad(self, v):
        # Gradient estimation steps from [-2,2] to [50,200]
        # return (((v + 2) / 4) * 250 + 50).astype(int)
        return ((v + 2) / 8) + 0.75 # to [0.75,1.25]
    
    def step(self, action):
        action = np.nan_to_num(action, nan=0.0, posinf=2, neginf=-2)
        self.reps += 1
        self.queries_left = self.steps - self.iter 
                    
        # Scale actions to proper values
        self.action_delta = self.scale_delta(action[0])*self.dist
        if self.reps == 1: self.action_delta = 0.1*self.dist
        self.action_step = self.scale_step(action[1])
        self.action_grad = self.scale_grad(action[2])
        num_grad = int(min([self.init_grad_evals * math.sqrt(self.reps), self.max_grad_evals]))
        self.action_grad = (num_grad * self.action_grad).astype(int)
        # self.action_binary = self.scale_binary(action[3])
        self.action_perlin = self.scale_perlin(action[3])
        self.action_binary = 1
        
        # Setting actions according to vanilla HSJA
        if self.nonadaptive:
            self.action_delta = self.select_delta(self.dist)
            # self.action_grad = int(min([self.init_grad_evals * math.sqrt(self.reps), self.max_grad_evals]))
            self.action_grad = num_grad
            self.action_step = 1/math.sqrt(self.reps)
            self.perlin = 0 
        
        # print(self.action_delta)
        # To force fixed number of queries, reduce gradient estimation steps if necessary
        self.action_grad = min(self.action_grad, max(self.queries_left-16, 16))
        # Gradient Estimation

        grad, self.mean_adv = self.approximate_gradients(self.best_advs, self.action_grad, self.action_delta, self.action_perlin)
            
        # Jump Step
        # !check if gradient is correct
        self.candidate, self.jump_steps = self.jump_step(grad, self.action_step, self.best_advs)
        # Binary Search
        self.best_advs, self.bin_steps = self.binary_step(ep.astensor(self.candidate), self.action_binary)
        
        # print(self.wanted_point.squeeze(0).numpy(), self.candidate.raw.squeeze(0).numpy())
        self.distance = l2(self.wanted_point, self.best_advs.raw.squeeze(0).numpy())
        # self.closer = self.distance < self.source_norm
        # print(action, self.lastStep)
        # print(self.closer, self.is_adv)

        # self.gain = l2(self.candidate.squeeze(0).numpy(), self.best_advs.raw.squeeze(0).numpy())
        # Calculate the distance to target gained in the last rep
        self.gain = self.dist - self.distance
        # print('gain:', self.gain)
        self.dist = self.distance
        self.gain_moving = self.gain_moving * 0.2 + (self.gain * 0.8) / self.gap
        # move inside functions
            # self.reward_mult = 1
            # self.improve_last += 1
            # self.gain = np.float32(0)
         
        # print(self.dist)
        # TODO: potentially reward shorter episodes       
        # self.converged = self.dist < self.goal
        if self.iter >= self.steps:
            self.tb.close()
            self.done = True
            print(self.dist)
        
        # if self.done:
        #     print(self.resets)
        #     print('finished in:', self.iter, 'steps and', self.reps, 'reps')
        #     print(self.dist,"|", self.gap)
            
        # print(self.action_delta, self.action_grad, self.action_step)    

        obs = self.observation()
        r = self.reward(self.rewarder)
        info = {"episode_number" : self.iter,
                "epsilon" : self.dist,
                "actions" : action,
                "correct" : True,
                "iters" : self.iter,
                "gap" : self.gap}
        return obs, r, self.done, info
    
    def observation(self):
        # generate observation based on the history of responses
        
        # History of success/fail, goal and/or distance to goal, (history of step sizes)
        
        # Use dist in place of moving dist
        loc = self.dist / self.gap
        self.dist_moving = self.dist_moving * 0.2 + (loc) * 0.8
        slope = self.dist_moving - loc
        # pos = self.best_advs.squeeze(0).numpy()
        # pos = pos[::4,::4].flatten().tolist()
        
        # Observation should also reflect the trajectory taken
        # by the binary search, grad approximation, and jump step
        observation = []
        # observation.append(1/self.bin_steps)
        observation.append((self.mean_adv+1)/2)
        observation.append(1/self.jump_steps)
        observation.append(loc)
        observation.append(slope)
        observation.append(self.gain/self.gap)
        # observation.extend(pos)
        # print(observation)
        # observation = np.append(observation, hist)
        # observation.append(self.gain)
        # observation.append(self.iter / self.steps)
        observation = np.nan_to_num(observation, nan=0.0, posinf=1, neginf=0)
        
        return observation

    def binary_step(self, best_advs, epsilon):
        # Adaptive: 1-shot attempt with given epsilon, non-adaptive: binary search
        # Try setting threshold instead; action_binary controls the dist where, 2-10
        # if self.nonadaptive:
        steps = 0
        # print(len(best_advs))
        highs = ep.ones(best_advs, len(best_advs))
        lows = ep.zeros_like(highs)
        threshold = 1 / self.dim ** 3
        best_candidate = best_advs
        
        while ep.any(highs - lows > threshold):
            mids = (lows + highs) / 2
            candidate = self.project(self.wanted_point, best_advs, mids)
            is_adv = self.is_adversarial(candidate.raw.unsqueeze(1))
            if is_adv:
                highs = mids
                best_candidate = candidate
            else:
                lows = mids
            steps += 1
            self.iter +=1
            # if self.iter >= self.steps - self.action_grad:
            #     break
        # print('bin steps:', steps)
        return best_candidate, steps
        # else:
        #     steps = 0
        #     while True:
        #         candidate = self.project(self.wanted_point, best_advs, epsilon)
        #         is_adv = self.is_adversarial(ep.astensor(candidate).raw.unsqueeze(1))
        #         steps += 1
        #         self.iter +=1
        #         if is_adv:
        #             best_candidate = candidate
        #             break
        #         elif steps >= 5:
        #             best_candidate = best_advs
        #         else:
        #             epsilon *= 1.5
        #             # print(epsilon)
        #     print('bin steps:', steps)
        #     return best_candidate, steps
        
    def approximate_gradients(self, x_advs, steps, delta, freq):
        if self.perlin:
            rv = pn.create_perlin_noise(x_advs.shape[-1], color=False, batch_size=steps, normalize=False, freq=freq)
            rv = ep.astensor(torch.tensor(rv).unsqueeze(1))
        else:
            noise_shape = tuple([steps] + list(x_advs.shape))
            # print(noise_shape)
            rv = ep.normal(x_advs, noise_shape)
        rv /= atleast_kd(ep.norms.l2(flatten(rv, keep=1), -1), rv.ndim) + 1e-12
        # scaled_rv = atleast_kd(ep.expand_dims(delta, 0), rv.ndim) * rv
        scaled_rv = delta * rv

        perturbed = ep.expand_dims(x_advs, 0) + scaled_rv
        perturbed = ep.clip(perturbed, 0, 1)

        rv = (perturbed - x_advs) / 2

        multipliers_list: List[ep.Tensor] = []
        for step in range(steps):
            decision = self.is_adversarial(perturbed[step].raw.unsqueeze(1))
            self.iter +=1
            multipliers_list.append(ep.ones(x_advs,1) if decision else -ep.ones(x_advs,1))
            #     ep.where(
            #         decision,
            #         ep.ones(x_advs, (len(x_advs,))),
            #         -ep.ones(x_advs, (len(x_advs,))),
            #     )
            # )
        # (steps, bs, ...)
        multipliers = ep.stack(multipliers_list, 0)
        # print(multipliers)
        
        vals = ep.where(
            ep.abs(ep.mean(multipliers, axis=0, keepdims=True)) == 1,
            multipliers,
            multipliers - ep.mean(multipliers, axis=0, keepdims=True),
        )
        grad = ep.mean(atleast_kd(vals, rv.ndim) * rv, axis=0)

        grad /= ep.norms.l2(atleast_kd(flatten(grad), grad.ndim)) + 1e-12
        # print('grad steps:', steps)
        return grad, ep.mean(multipliers, axis=0).raw.squeeze(0).numpy()
    
    def perlin_gradients(self, x_advs, steps, delta, freq):
        # noise_shape = tuple([steps] + list(x_advs.shape))
        rv = pn.create_perlin_noise(x_advs.shape[-1], color=False, batch_size=steps, normalize=False, freq=freq)
        rv = ep.astensor(torch.tensor(rv).unsqueeze(1))
        # print(rv.shape)
        # rv = ep.normal(x_advs, noise_shape)
        rv /= atleast_kd(ep.norms.l2(flatten(rv, keep=1), -1), rv.ndim) + 1e-12
        # scaled_rv = atleast_kd(ep.expand_dims(delta, 0), rv.ndim) * rv
        scaled_rv = delta * rv

        perturbed = ep.expand_dims(x_advs, 0) + scaled_rv
        perturbed = ep.clip(perturbed, 0, 1)

        rv = (perturbed - x_advs) / 2

        multipliers_list: List[ep.Tensor] = []
        for step in range(steps):
            decision = self.is_adversarial(perturbed[step].raw.unsqueeze(1))
            self.iter +=1
            multipliers_list.append(ep.ones(x_advs,1) if decision else -ep.ones(x_advs,1))
            #     ep.where(
            #         decision,
            #         ep.ones(x_advs, (len(x_advs,))),
            #         -ep.ones(x_advs, (len(x_advs,))),
            #     )
            # )
        # (steps, bs, ...)
        multipliers = ep.stack(multipliers_list, 0)
        # print(multipliers)
        
        vals = ep.where(
            ep.abs(ep.mean(multipliers, axis=0, keepdims=True)) == 1,
            multipliers,
            multipliers - ep.mean(multipliers, axis=0, keepdims=True),
        )
        grad = ep.mean(atleast_kd(vals, rv.ndim) * rv, axis=0)

        grad /= ep.norms.l2(atleast_kd(flatten(grad), grad.ndim)) + 1e-12
        # print('grad steps:', steps)
        return grad, ep.mean(multipliers, axis=0).raw.squeeze(0).numpy()
    
    def jump_step(self, grad, step, x_advs):
        steps = 0
        epsilon = self.dist * step
        # print(grad)
        while True:
            # candidate = ep.clip(x_advs + atleast_kd(epsilon, x_advs.ndim) * grad, 0, 1)
            candidate = ep.clip(x_advs + epsilon * grad, 0, 1)
            success = self.is_adversarial(candidate.raw.unsqueeze(1))
            steps += 1
            self.iter += 1
            if success:
                break
            else:
                epsilon /= 2
        # print('jump steps:', steps)
        return candidate, steps
        
    def project(self, originals, perturbed, epsilons):
        return (1.0 - epsilons) * originals + epsilons * perturbed
    
    def select_delta(self, dist):
        if self.reps == 1:
            result = 0.1 * dist
        else:
            theta = 1 / (self.dim ** 2)
            result = theta * self.dist 
        return result

    # =============================================================

    # def reward1(self):
    #     reward = self.gain / self.gap
    #     return reward
    
    def reward1(self):
        reward = self.gain*10
        return reward

    # def reward2(self):
    #     fraction = self.dist / self.gap
    #     fraction_previous = (self.dist + self.gain) / self.gap
    #     reward = (1 - fraction ** 0.5) ** 2 - (1 - fraction_previous ** 0.5) ** 2
    #     return reward
    
    def reward2(self):
        reward = 10/self.action_grad + 0.1/self.jump_steps + self.reward1()
        return reward

    def reward3(self):
        fraction = self.dist / self.gap
        reward = (1 - fraction ** 2) ** 0.5
        return reward*10

    def reward4(self):
        reward = 10/self.action_grad + self.reward1()
        return reward
    
    # def reward5(self):
    #     reward = 0
    #     if self.iter >= self.steps:
    #         reward = abs(math.log(self.dist / self.gap))
    #     return reward
    
    def reward5(self):
        reward = -self.action_grad/200 + self.reward1()
        return reward

    # def reward5(self):
    #     reward = 0
    #     if self.iter >= self.steps:
    #         reward = abs(math.log(self.dist / self.gap))
    #     return reward

    def reward(self, reward_nr):
        if reward_nr == 1:
            # R1
            reward = self.reward1()
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
        self.tb.scalar("reward", torch.tensor([reward]), self.iter)
        return reward

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