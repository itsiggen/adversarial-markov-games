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
import utils.pnoise as pn
from collections import deque
from models.trainMNISTtorch import Net
import math

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class BagsSkip(gym.Env):
    def __init__(
        self,
        steps: int = 1000,
        spherical_step: float = 1e-2,
        source_step: float = 1e-2,
        defended = False,
        nonadaptive = False,
        train = True,
        rewarder = 1,
        scale = 25,
        dataset = None,
        seed = 2,
        tensorboard = False
        ):
        super(BagsSkip, self).__init__()  

        # Boundary Attack inits
        self.steps = steps
        self.spherical_step = spherical_step
        self.source_step = source_step
        self.nonadaptive = nonadaptive
        self.rewarder = rewarder
        self.scale = scale
        self.tensorboard = tensorboard
        random.seed(seed)

        # Actions space
        self.action_space = spaces.Box(low=-2, high=2, shape=(4,), dtype=np.float32)
        # Observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)

        # Load MNIST pytorch CNN model -- 99.1% acc -- 98.6% acc adversarially trained
        self.dataset = dataset
        self.mode = Net()
        if defended:
            self.mode.load_state_dict(torch.load('./models/mnist_cnn_adv.pt', map_location=device))
        else:
            self.mode.load_state_dict(torch.load('./models/mnist_cnn.pt', map_location=device))
        self.mode.eval()

        self.model = PyTorchModel(self.mode, bounds=(0, 1), device=device)
        self.indices = [0,7999] if train else [8000,9999]
        self.dim = 28
        self.resets = 0
    
        self.done = False

    def scale_perlin(self, v):
        act = ((v + 2) / 4) * (self.dim - 2) + 1
        return np.nan_to_num(act, nan=0.0, posinf=self.dim-1, neginf=0.0)

    def scale_mask(self, v):
        return (v + 2) / 2

    def scale_step(self, v):
        return (v + 2) / self.scale
    
    def reset(self):
        # Initialize new targeted attack
        self.iter = 0
        self.tb = TensorBoard(logdir=self.tensorboard)
        self.starting_point, startLabel, self.wanted_point, originLabel = self.get_pair()
        # self.starting_point, _ = ep.astensor_(self.starting_point)
        # self.original, self.restore_type = ep.astensor_(self.wanted_point)
        # if self.resets < 3: print("Start:", startLabel, "| Wanted:", originLabel)
        self.resets += 1
        self.criterion = TargetedMisclassification(torch.tensor([startLabel]))
        # Distance between starting and origin point / current best adv
        self.gap = l2(self.starting_point, self.wanted_point)
        self.dist = self.gap
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
        # self.epsilon = 1
        self.actions = []
        self.done = False
        self.success = False

        self.is_adversarial = get_is_adversarial(self.criterion, self.model)
        
        if self.starting_point is None:
            raise ValueError("no starting_point provided")
        else:
            self.best_advs = self.starting_point

        is_adv  = self.is_adversarial(ep.astensor(torch.tensor(self.best_advs).unsqueeze(0).unsqueeze(1)))
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
        if self.iter >= self.steps:
            self.tb.close()
            self.done = True
        
        # Scale actions to proper values
        self.action_perlin = self.scale_perlin(action[0])
        self.action_mask = self.scale_mask(action[1])
        self.action_spherical = self.scale_step(action[2])
        self.action_source = self.scale_step(action[3])

        # calculate mask
        mask = np.abs(self.best_advs - self.wanted_point)
        mask /= np.max(mask)
        self.x_mask = mask

        # # calc step sizes
        # source_step = 0.002
        # spherical_step = 0.05
        # if self.step_loop_current >= self.step_loop_max:
        #     self.step_loop_current = 0
        # scale = (1. - self.step_loop_current / self.step_loop_max) + 0.3
        # source_step_size = source_step * scale
        # spherical_step_size = spherical_step * scale
        # self.step_loop_current += 1
        
        # Setting actions according to vanilla BAGS    
        if self.nonadaptive:
            scale = (1. - min(self.na_batch/50, 1)) + 0.3
            self.action_perlin = 5
            self.action_mask = 1
            self.action_spherical = scale * self.spherical_step
            self.action_source = scale * self.source_step
        
        # generate new advarsarial candidate
        self.candidate = self.generate_boundary_sample(self.wanted_point, self.best_advs, self.x_mask, self.action_source,
                                                     self.action_spherical, self.action_perlin)
        self.is_adv = self.is_adversarial(ep.astensor(torch.tensor(self.candidate).unsqueeze(0).unsqueeze(1)))
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
            # self.dist = l2(self.best_advs, self.wanted_point)
            self.dist = self.distance
            self.gain_moving = self.gain_moving * 0.8 + (self.gain * 0.2) / self.gap
            self.improve_avg = self.improve_avg * 0.8 + (1/(self.improve_last +1))*0.2
            self.reward_mult = self.improve_last
            self.improve_last = 0
            self.na_batch = 1
            # print(self.dist, self.iter)
            
            self.unnormalized_source_direction = self.wanted_point - self.best_advs
            # self.source_norm = np.linalg.norm(self.unnormalized_source_direction)
            self.source_norm = self.distance
            self.source_direction = self.unnormalized_source_direction / self.source_norm
            
        else:
            self.reward_mult = 1
            self.improve_last += 1
            self.na_batch += 1
            # nonadaptive batch
            if self.improve_last >= 49:
                self.na_batch = 1
            self.gain = np.float32(0)
        
        # # TODO: potentially reward shorter episodes
        # is_within_eps = self.dist < self.epsilon # check if perturbation < eps    
        # if is_best_adv and is_within_eps:
        #     self.done = True
        #     self.success = True
        #     # print('success')
        
        # self.unnormalized_source_direction = self.wanted_point - self.best_advs
        # self.source_norm = np.linalg.norm(self.unnormalized_source_direction)
        # self.source_direction = self.unnormalized_source_direction / self.source_norm
        # update tensorboard
        # self.update_tb(is_best_adv, cond

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
        # observation.extend(pos)
        # observation = np.append(observation, hist)
        # observation.append(self.gain)
        observation.append(self.iter / 5000)
        observation.append(self.gap/15)
        observation.append(self.dist/15)
        # print(observation)
        
        return observation

    def generate_boundary_sample(self, X_orig, X_adv_current, mask, source_step, spherical_step, perlin_freq):
        # Adapted from FoolBox BoundaryAttack.
            
        mask = mask ** self.action_mask
        # rnd_normal = pn.create_perlin_noise(self.dim, color=False, freq=self.action_perlin, normalize=False).squeeze(0)
        rnd_normal = pn.generate_perlin_noise_2d(self.dim, perlin_freq)
        rnd_normal /= np.linalg.norm(rnd_normal)
        sampling_dir = rnd_normal

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
        return reward

    def reward2(self):
        if self.gain > 0:
            reward = (self.gain / self.gap) / (self.reward_mult + 1)
        else:
            reward = 0
        return reward
    
    def reward3(self):
        fraction = self.dist / self.gap
        fraction_previous = (self.dist + self.gain) / self.gap
        reward = (1 - fraction ** 0.5) ** 2 - (1 - fraction_previous ** 0.5) ** 2
        return reward

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
        
        return startImg.squeeze(0).numpy(), startLabel, originImg.squeeze(0).numpy(), originLabel