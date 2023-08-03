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
from utils.queues import l2
import utils.perlin as pn
from torchvision import transforms
from collections import deque, OrderedDict
from models.trainCIFARtorch import resnet20
import math
import pandas as pd

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
np.seterr(invalid='raise')

class BagsSkipCIFAR(gym.Env):
    def __init__(
        self,
        steps: int = 1000,
        spherical_step: float = 1e-2,
        source_step: float = 1e-2,
        defended = False,
        nonadaptive = False,
        train = True,
        test = False,
        rewarder = 1,
        scale = 5,
        dataset = None,
        tensorboard = False
        ):
        super(BagsSkipCIFAR, self).__init__()  

        # Boundary Attack inits
        self.steps = steps
        self.spherical_step = spherical_step
        self.source_step = source_step
        self.nonadaptive = nonadaptive
        self.rewarder = rewarder
        self.train = train
        self.test = test
        self.scale = scale
        self.pairs = pd.read_csv('utils/pairs.csv').to_numpy()

        # Actions space
        self.action_space = spaces.Box(low=-2, high=2, shape=(4,), dtype=np.float32)
        # Observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)

        # Load CIFAR pytorch Resnet20 model -- 92.1% acc -- 88.25% acc adversarially trained
        self.dataset = dataset

        model = resnet20()
        if defended:
            model.load_state_dict(torch.load('./models/cifar_resnet_adv.pt', map_location=device)['state_dict'])
        else:
            model.load_state_dict(torch.load('./models/cifar_resnet.pt', map_location=device)['state_dict'])
        model.eval()
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.model = PyTorchModel(model, bounds=(0, 1), device=device)
        self.indices = [0,7999] if train else [8000,9999]
        self.dim = 32
        self.channels = 3
        self.resets = 0
        self.done = False

    def scale_perlin(self, v):
        act = ((v + 2) / 4) * (self.dim - 3) + 1
        return np.nan_to_num(act, nan=0.0, posinf=self.dim-2, neginf=0.0)

    def scale_mask(self, v):
        return (v + 2) / 2

    def scale_step(self, v):
        return (v + 2) / self.scale
    
    def reset(self):
        # Initialize new targeted attack
        self.iter = 0
        self.starting_point, startLabel, self.wanted_point, originLabel = self.get_pair()
        # self.starting_point, _ = ep.astensor_(self.starting_point)
        # self.original, self.restore_type = ep.astensor_(self.wanted_point)
        # if self.resets < 1: print("Start:", startLabel, "| Wanted:", originLabel)
        # print("Start:", startLabel, "| Wanted:", originLabel)
        self.resets += 1
        # if self.resets == 100:
        #     df = pd.DataFrame(self.pairs)
        #     df.to_csv('pairs.csv', index=False)
        self.criterion = TargetedMisclassification(torch.tensor([startLabel]))
        # Distance between starting and origin point / current best adv
        self.gap = l2(self.starting_point, self.wanted_point)
        # print("Start:", startLabel, "| Wanted:", originLabel, "| Gap:", self.gap)
        self.dist = self.gap
        # print(self.dist)
        # Distance between successive steps
        self.diff = np.float32(0.0)
        # Moving average of the closing distance
        self.dist_moving = np.float32(1.0)
        # Initial mask
        self.x_mask = np.ones(self.wanted_point.shape, dtype=np.float32)
        # Last iter for improvement
        self.improve_last = 0
        self.na_batch = 1
        self.improve_avg = 1
        self.reward_mult = 1
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
            self.best_advs = self.starting_point
        
        cand = self.normalize(torch.tensor(self.best_advs))
        is_adv = self.is_adversarial(ep.astensor(cand.unsqueeze(0)))
        if not is_adv:
            raise ValueError("starting_point is not adversarial")

        self.bounds = self.model.bounds
        self.unnormalized_source_direction = self.wanted_point - self.best_advs
        self.source_norm = np.linalg.norm(self.unnormalized_source_direction)
        self.source_direction = self.unnormalized_source_direction / self.source_norm
    
        # create queues to track various statistic used to derive the state
        # success rate, step size, relative location, progress in episode
        self.stats_is_adv = deque(maxlen=30)
        self.dist_derivative = deque(maxlen=30)
        self.improve_time_avg = deque(maxlen=30)
        self.moving_avg_step_dist = deque(maxlen=30)
        # pos = self.best_advs[::4,::4].flatten()
    
        observation = []
        observation.append(np.float32(0.0))
        observation.append(np.float32(1.0))
        observation.append(slope)
        observation.append(self.improve_avg)
        observation.append(self.gain_moving)
        ### EXTRA
        observation.append(self.iter / 5000)
        observation.append(self.gap/15)
        observation.append(self.dist/15)
        # observation.extend(pos)
        # observation = np.append(observation, np.ones(30))

        return observation

    def step(self, action):
        if np.isnan(action).any():
            print("nan")
        self.iter += 1
        
        # self.converged = self.dist < self.epsilon
        # if self.converged or self.iter >= self.steps:
        if self.iter >= self.steps:
            self.done = True
            # print(self.dist)
        
        # Scale actions to proper values
        self.action_perlin = self.scale_perlin(action[0])
        self.action_mask = self.scale_mask(action[1])
        self.action_spherical = self.scale_step(action[2])
        self.action_source = self.scale_step(action[3])

        # calculate mask
        # if self.iter == 1 or self.iter % 10 == 0:
        mask = np.abs(self.best_advs - self.wanted_point)
        mask /= np.max(mask)
        self.x_mask = mask
        
        # Setting actions according to vanilla BAGS    
        if self.nonadaptive:
            # check
            scale = (1. - min(self.na_batch/50, 1)) + 0.3
            # print(scale)
            self.action_perlin = 5
            self.action_mask = 0.5
            self.action_spherical = scale * self.spherical_step
            self.action_source = scale * self.source_step
        
        # generate new advarsarial candidate
        self.candidate = self.generate_boundary_sample(self.wanted_point, self.best_advs, self.x_mask, self.action_source,
                                                     self.action_spherical, self.action_perlin)
        cand = self.normalize(torch.tensor(self.candidate))
        self.is_adv = self.is_adversarial(ep.astensor(cand.unsqueeze(0)))
        self.stats_is_adv.append(self.is_adv.numpy()[0])
        
        self.distance = l2(self.wanted_point, self.candidate)
        self.closer = self.distance < self.source_norm
        # print(action, self.lastStep)
        # print(self.closer, self.is_adv)
        is_best_adv = self.is_adv and self.closer
        # print(is_best_adv)

        if is_best_adv:
            # self.gain = l2(self.candidate, self.best_advs)
            self.gain = self.source_norm - self.distance
            self.best_advs = self.candidate
            self.dist = l2(self.best_advs, self.wanted_point)
            self.gain_moving = self.gain_moving * 0.8 + (self.gain * 0.2) / self.gap
            self.improve_avg = self.improve_avg * 0.8 + (1/(self.improve_last +1))*0.2
            self.reward_mult = self.improve_last
            self.improve_last = 0
            self.na_batch = 1
            # print(self.dist, self.iter)
        else:
            self.reward_mult = 1
            self.improve_last += 1
            self.na_batch += 1
            # nonadaptive batch
            if self.improve_last >= 49:
                self.na_batch = 1
            self.gain = np.float32(0)
        
        # TODO: potentially reward shorter episodes
        # is_within_eps = self.dist < self.epsilon # check if perturbation < eps    
        # if is_best_adv and is_within_eps:
        #     self.done = True
        #     self.success = True
        #     print('success')
        
        self.unnormalized_source_direction = self.wanted_point - self.best_advs
        self.source_norm = np.linalg.norm(self.unnormalized_source_direction)
        self.source_direction = self.unnormalized_source_direction / self.source_norm

        obs = self.observation()
        if np.isnan(obs).any():
            print("nan")
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
        self.dist_moving = self.dist_moving * 0.8 + (self.dist / self.gap) * 0.2
        loc = self.dist / self.gap
        slope = self.dist_moving - loc
        

        observation = []
        # hist = np.asarray(self.stats_is_adv, dtype=int)
        # if len(hist) < 30:
        #     hist = np.ones(30)
        # print(np.asarray(self.stats_is_adv, dtype=int))
        
        # pos = self.best_advs[::4,::4].flatten().tolist()
        
        observation.append(np.mean(self.stats_is_adv))
        observation.append(loc)
        observation.append(slope)
        observation.append(self.improve_avg)
        observation.append(self.gain_moving)
        ### EXTRA
        observation.append(self.iter / 5000)
        observation.append(self.gap/15)
        observation.append(self.dist/15)
        # observation.extend(pos)
        # observation = np.append(observation, hist)
        # observation.append(self.gain)
        # observation.append(self.iter / self.steps)
        # print(observation)
        
        return observation

    def generate_boundary_sample(self, X_orig, X_adv_current, mask, source_step, spherical_step, perlin_freq):
        # Adapted from FoolBox BoundaryAttack.
            
        mask = mask ** self.action_mask
        # rnd_normal = pn.create_perlin_noise(self.dim, color=False, freq=self.action_perlin, normalize=False).squeeze(0)
        # rnd_normal = pn.generate_perlin_noise_2d(self.dim, perlin_freq)
        # rnd_normal /= np.linalg.norm(rnd_normal)
        # rnd_normal1 = pn.generate_perlin_noise_2d(self.dim, perlin_freq)
        # rnd_normal1 /= np.linalg.norm(rnd_normal1)
        # rnd_normal2 = pn.generate_perlin_noise_2d(self.dim, perlin_freq)
        # rnd_normal2 /= np.linalg.norm(rnd_normal2)
        # # triple stack noise tensor
        # sampling_dir = np.stack((rnd_normal, rnd_normal1, rnd_normal2))
        sampling_dir = np.squeeze(pn.create_perlin_noise(self.dim, freq=perlin_freq))

        # calculate candidate on sphere
        dot = np.vdot(sampling_dir, self.source_direction)
        sampling_dir -= dot * self.source_direction  # Project orthogonal to source direction
        sampling_dir *= mask  # Apply regional mask
        sampling_dir /= np.linalg.norm(sampling_dir)  # Norming increases magnitude of masked regions

        sampling_dir *= spherical_step * self.source_norm  # Norm to length stepsize*(dist from src)

        D = 1 / np.sqrt(spherical_step ** 2 + 1)
        direction = sampling_dir - self.unnormalized_source_direction
        spherical_candidate = X_orig + D * direction

        np.clip(spherical_candidate, 0., 1., out=spherical_candidate)

        # step towards source
        new_source_direction = X_orig - spherical_candidate

        new_source_direction_norm = np.linalg.norm(new_source_direction)
        new_source_direction /= new_source_direction_norm
        spherical_candidate = X_orig - self.source_norm * new_source_direction  # Snap sph.c. onto sphere

        # From there, take a step towards the target.
        candidate = spherical_candidate + (source_step * self.source_norm) * new_source_direction

        np.clip(candidate, 0., 1., out=candidate)
        return np.float32(candidate)

    def reward1(self):
        if self.gain > 0:
            reward = (self.gain / self.gap) * self.reward_mult
        else:
            reward = 0
        return reward*10

    def reward2(self):
        if self.gain > 0:
            reward = (self.gain / self.gap) / (self.reward_mult + 1)
        else:
            reward = 0
        return reward*10
    
    def reward3(self):
        fraction = self.dist / self.gap
        fraction_previous = (self.dist + self.gain) / self.gap
        reward = (1 - fraction ** 0.5) ** 2 - (1 - fraction_previous ** 0.5) ** 2
        return reward*10

    def reward4(self):
        reward = math.sqrt(self.iter) * self.reward2()
        return reward

    def reward5(self):
        reward = 0
        if self.iter >= self.steps:
            reward = abs(math.log(self.dist / self.gap))
        return reward

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
        return reward

    def get_pair(self):
        if self.test:
            startImgNr = self.pairs[self.resets][0]
            originImgNr = self.pairs[self.resets][1]
        else:
            startImgNr = random.randint(*self.indices)
            originImgNr = random.randint(*self.indices)
            
            # Make sure original image is correctly classified by the model
            while not ep.argmax(self.model(self.normalize(self.dataset[originImgNr][0]).unsqueeze(0))).detach().numpy() == self.dataset[originImgNr][1]:
                originImgNr = random.randint(*self.indices)
            
            # Make sure starting and original images do not belong to the same class, and starting is correctly classified
            while self.dataset[startImgNr][1] == self.dataset[originImgNr][1] \
                or not ep.argmax(self.model(self.normalize(self.dataset[startImgNr][0]).unsqueeze(0))).detach().numpy() == self.dataset[startImgNr][1]:
                startImgNr = random.randint(*self.indices)
            
        startImg = self.dataset[startImgNr][0].to(device)
        startLabel = self.dataset[startImgNr][1]
        
        originImg = self.dataset[originImgNr][0].to(device)
        originLabel = self.dataset[originImgNr][1]
        
        # self.pairs.append([startImgNr, originImgNr])
    
        return startImg.squeeze(0).numpy(), startLabel, originImg.squeeze(0).numpy(), originLabel