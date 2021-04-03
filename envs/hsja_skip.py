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
from utils.buckets import l2
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
        dataset = None,
        seed = 2,
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
        self.tensorboard = tensorboard
        random.seed(seed)

        # Actions space
        self.action_space = spaces.Box(low=-2, high=2, shape=(4,), dtype=np.float32)
        # Observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(54,), dtype=np.float32)

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
    
        self.done = False
    
    def reset(self):
        # Initialize new targeted attack 
        self.iter = 0
        self.reps = 0
        self.tb = TensorBoard(logdir=self.tensorboard)
        self.starting_point, startLabel, self.wanted_point, originLabel = self.get_pair()
        # self.starting_point, _ = ep.astensor_(self.starting_point)
        # self.original, self.restore_type = ep.astensor_(self.wanted_point)
        # print("Start:", startLabel, "| Wanted:", originLabel)
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

        self.is_adversarial = get_is_adversarial(self.criterion, self.model)
        
        if self.starting_point is None:
            raise ValueError("no starting_point provided")
        else:
            self.best_advs = self.starting_point

        is_adv  = self.is_adversarial(ep.astensor(self.best_advs).raw.unsqueeze(1))
        if not is_adv:
            raise ValueError("starting_point is not adversarial")
            
        self.best_advs, self.extra_queries = self.binary_step(ep.astensor(self.best_advs), 1)    
        pos = self.best_advs.squeeze(0).numpy()
        pos = pos[::4,::4].flatten().tolist()
        
        observation = []
        # observation.append(np.float32(1.0))
        # TODO: encode source n target class in obs
        observation.append(np.float32(0.5))
        observation.append(np.float32(1.0))
        observation.append(np.float32(1.0))
        observation.append(np.float32(0.5))
        observation.append(np.float32(0.0))
        observation.extend(pos)

        return observation
    
    def step(self, action):
        self.reps += 1
        self.queries_left = self.steps - self.iter 
                    
        # Scale actions to proper values
        self.action_binary = self.scale_binary(action[0])
        self.action_delta = self.scale_delta(action[1])
        self.action_step = self.scale_step(action[2])
        self.action_grad = self.scale_grad(action[3])
        
        # Setting actions according to vanilla HSJA
        if self.nonadaptive:
            self.action_delta = self.select_delta(ep.astensor(torch.tensor(self.dist)))
            self.action_grad = int(min([self.init_grad_evals * math.sqrt(self.reps + 1), self.max_grad_evals]))
            self.action_step = 1/math.sqrt(self.reps + 1)
        
        # print(self.action_delta)
        # To force fixed number of queries, reduce gradient estimation steps if necessary
        self.action_grad = min(self.action_grad, max(self.queries_left-16, 16))
        # Gradient Estimation
        grad, self.mean_adv = self.approximate_gradients(self.best_advs, self.action_grad, self.action_delta)
        # Jump Step
        # !check if gradient is correct
        self.candidate, self.jump_steps = self.jump_step(grad, self.action_step, self.best_advs)
        # Binary Search
        self.best_advs, self.bin_steps = self.binary_step(ep.astensor(self.candidate), self.action_binary)
        
        # print(self.wanted_point.squeeze(0).numpy(), self.candidate.raw.squeeze(0).numpy())
        self.distance = l2(self.wanted_point.squeeze(0).numpy(), self.best_advs.raw.squeeze(0).numpy())
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
        
        # TODO: potentianlly reward shorter episodes
        is_within_eps = self.dist < self.goal
        if is_within_eps:
            self.done = True
            self.success = True
            # print('success')

        # update tensorboard
        # self.update_tb(is_best_adv, cond
        
        # print(self.dist)
        
        self.converged = self.dist < self.epsilon
        if self.converged or self.iter >= self.steps:
            self.tb.close()
            self.done = True
        
        # if self.done:
        #     print('finished in:', self.iter, 'steps and', self.reps, 'reps')
        #     print(self.dist,"|", self.gap)

        obs = self.observation()
        r = self.reward(self.rewarder)
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
        loc = self.dist / self.gap
        self.dist_moving = self.dist_moving * 0.2 + (loc) * 0.8
        slope = self.dist_moving - loc
        pos = self.best_advs.squeeze(0).numpy()
        pos = pos[::4,::4].flatten().tolist()
        
        # Observation should also reflect the trajectory taken
        # by the binary search, grad approximation, and jump step
        observation = []
        # observation.append(1/self.bin_steps)
        observation.append((self.mean_adv+1)/2)
        observation.append(1/self.jump_steps)
        observation.append(loc)
        observation.append(slope)
        observation.append(self.gain/self.gap)
        observation.extend(pos)
        # print(observation)
        # observation = np.append(observation, hist)
        # observation.append(self.gain)
        # observation.append(self.iter / self.steps)
        
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
            is_adv = self.is_adversarial(ep.astensor(candidate).raw.unsqueeze(1))
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
        #         else:
        #             epsilon *= 1.5
        #             # print(epsilon)
        #     print('bin steps:', steps)
        #     return best_candidate, steps
        
    def approximate_gradients(self, x_advs, steps, delta):
        noise_shape = tuple([steps] + list(x_advs.shape))
        rv = ep.normal(x_advs, noise_shape)            
        rv /= atleast_kd(ep.norms.l2(flatten(rv, keep=1), -1), rv.ndim) + 1e-12
        # scaled_rv = atleast_kd(ep.expand_dims(delta, 0), rv.ndim) * rv
        scaled_rv = delta * rv

        perturbed = ep.expand_dims(x_advs, 0) + scaled_rv
        perturbed = ep.clip(perturbed, 0, 1)

        rv = (perturbed - x_advs) / 2

        multipliers_list: List[ep.Tensor] = []
        for step in range(steps):
            decision = self.is_adversarial(ep.astensor(perturbed[step]).raw.unsqueeze(1))
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
            success = self.is_adversarial(ep.astensor(candidate).raw.unsqueeze(1))
            steps += 1
            self.iter += 1
            if success:
                break
            else:
                epsilon /= 2
        # print('jump steps:', steps)
        return candidate, steps
    
    def scale_binary(self, v):
        # Binary search parameter from [-2,2] to [0.01, 0.41]
        return (v + 2) / 8 + 0.01
    
    def scale_delta(self, v):
        # Delta from [-2,2] to [0,0.1]
        return (v + 2) / 40
    
    def scale_step(self, v):
        # Jump step search from [-2,2] to [0.1,1]
        return (v + 2) / 4 + 0.1
    
    def scale_grad(self, v):
        # Gradient estimation steps from [-2,2] to [100,1000]
        return (((v + 2) / 4) * 300 + 100).astype(int)
        
    def project(self, originals, perturbed, epsilons):
        return (1.0 - epsilons) * originals + epsilons * perturbed
    
    def select_delta(self, dist):
        if self.reps == 1:
            result = 0.1 * ep.ones_like(dist)
        else:
            theta = 1 / (self.dim ** 2)
            result = theta * self.dist 
        return result

    # =============================================================

    def reward1(self):
        reward = self.gain / self.gap
        return reward

    def reward2(self):
        fraction = self.dist / self.gap
        fraction_previous = (self.dist + self.gain) / self.gap
        reward = (1 - fraction ** 0.5) ** 2 - (1 - fraction_previous ** 0.5) ** 2
        return reward

    def reward3(self):
        fraction = self.dist / self.gap
        reward = (1 - fraction ** 2) ** 0.5
        return reward

    def reward4(self):
        reward = 1/self.jump_steps + self.reward1()
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