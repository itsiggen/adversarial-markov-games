import eagerpy as ep
import numpy as np
import gym
import torch
import random
from foolbox import PyTorchModel
from foolbox.tensorboard import TensorBoard
# from foolbox.attacks.base import get_is_adversarial
from gym import spaces
from foolbox.criteria import TargetedMisclassification
from utils.utils import flatten, atleast_kd
from utils.queues import Queues, l2
import utils.pnoise as pn
from utils.utils import get_is_adversarial
from models.trainMNISTtorch import Net
from models.trainAdvMNISTtorch import LeNet5
from collections import deque
import matplotlib.pyplot as plt
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.seterr(invalid='raise')

class BagsGames(gym.Env):
    def __init__(
        self,
        steps: int = 1000,
        spherical_step: float = 1e-2,
        source_step: float = 1e-2,
        defended = False,
        adaptive: int = 0,
        ratio_benign = 0.5,
        train = True,
        rewarder = 1,
        dataset = None,
        seed = 2,
        tensorboard = False,
        update_stats_every_k: int = 10
        ):
        super(BagsGames, self).__init__()  

        # Boundary Attack inits
        self.steps = steps
        self.spherical_step = spherical_step
        self.source_step = source_step
        self.adaptive = adaptive  # 0: none adaptive | 1: adv adaptive | 2: int adaptive | 3: both adaptive
        self.ratio_benign = ratio_benign
        self.rewarder = rewarder
        self.tensorboard = tensorboard
        random.seed(seed)

        # Observation space
        self.observation_spaces = spaces.Dict({
            'adversary': spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32),
            'interceptor': spaces.Box(low=0, high=1, shape=(300,), dtype=np.float32)
            })

        # Actions space
        self.action_spaces = spaces.Dict({
            'adversary': spaces.Box(low=-2, high=2, shape=(4,), dtype=np.float32),
            'interceptor': spaces.Box(low=-2, high=2, shape=(1,), dtype=np.float32)
            })
        
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

    def scale_perlin(self, v):
        return ((v + 2) / 4) * (self.dim - 2) + 1

    def scale_mask(self, v):
        return (v + 2) / 2

    def scale_step(self, v):
        return (v + 2) / 20
    
    def scale_intercept(self, v):
        return (v + 2) / 4
    
    def reset(self):
        """ Initialize new targeted attack
        """
        self.iter = 0           # num of attack queries
        self.queries = 0        # num of benign queries
        self.tb = TensorBoard(logdir=self.tensorboard)
        self.starting_point, startLabel, self.wanted_point, originLabel = self.get_pair()
        # self.starting_point, _ = ep.astensor_(self.starting_point)
        # self.original, self.restore_type = ep.astensor_(self.wanted_point)
        if self.resets < 5: print("Start:", startLabel, "| Wanted:", originLabel)
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
        self.improve_avg = 1
        self.reward_mult = 1
        # Moving average of the step
        self.step_moving = 0.1
        self.gain_moving = 0.1
        slope = 0.5
        # Initialize query queues
        self.queues = Queues(nrQueues=2)
        # Target epsilon
        self.epsilon = 1
        self.correct = []
        self.done = False
        self.success = False
        # Set current and next player
        self.curr = 1
        self.next = 0

        self.is_adversarial = get_is_adversarial(self.criterion, self.model)
        
        if self.starting_point is None:
            raise ValueError("no starting_point provided")
        else:
            self.best_advs = self.starting_point

        self.is_adv, self.logits = self.is_adversarial(ep.astensor(torch.tensor(self.best_advs).unsqueeze(0).unsqueeze(1)))
        self.is_adv = self.is_adv.raw.numpy()[0]
        if not self.is_adv:
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
        
        self.candidate = self.best_advs
    
        # Return observation for interceptor as it's the first agent to move
        obs, ix, self.lastStep, self.alt = self.observation_int(self.logits, self.best_advs)
        
        # observation = []
        # observation.append(np.float32(0.0))
        # observation.append(np.float32(1.0))
        # observation.append(slope)
        # observation.append(self.improve_avg)
        # observation.append(self.gain_moving)
        # observation.extend(pos)
        # observation = np.append(observation, np.ones(30))

        return obs

    def step(self, action):
        """
        Progress through the internal states of the environment: interceptor 
        always follows after adversary or benign, and adversary or benign follows
        after interceptor based on a predefined probability
        """
        if self.curr == 1 or self.curr == 2:
            # Int responds to adv or ben
            self.past = self.curr
            obs, r, done, info = self.step_int(action)
            self.roll_next()
            return obs, r, done, info, 0, self.next
        elif self.curr == 0:
            if self.next == 1:
                # Adv follows int
                obs, r, done, info = self.step_adv(action)
            elif self.next == 2:
                # Ben follows int
                obs, r, done, info = self.step_ben(action)
            return obs, r, done, info, self.curr, 0
            
    def step_int(self, action):
        if self.curr == 1:
            # Candidate remains adversarial only if outside the containment area
            candid = self.switch(self.lastStep, action)
            self.is_adv = np.logical_and(self.is_adv, candid)
          
            self.stats_is_adv.append(self.is_adv[0])
            
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
            else:
                self.reward_mult = 1
                self.improve_last += 1
                self.gain = np.float32(0)

            is_within_eps = self.dist < self.epsilon # check if perturbation < eps    
            if is_best_adv and is_within_eps:
                self.done = True
                self.success = True
                # print('success')
            
            self.unnormalized_source_direction = self.wanted_point - self.best_advs
            self.source_norm = np.linalg.norm(self.unnormalized_source_direction)
            self.source_direction = self.unnormalized_source_direction / self.source_norm
            # update tensorboard
            # self.update_tb(is_best_adv, cond
    
            # Make obs object var so it can be returned from benign
            self.obs = self.observation_adv()

            r = self.reward_adv(self.rewarder)
            info = self.get_info()
            # Set state to interceptor
            self.curr = 0
            return self.obs, r, self.done, info
        
        elif self.curr == 2:
            # Classify benign input
            if self.switch(self.lastStep, action):
                # print(np.argsort(torch.nn.functional.softmax(self.logits[0]))[-1])
                ans = np.argsort(torch.nn.functional.softmax(self.logits, dim=1))[0][-1]
            else:
                ans = self.alt
            # Check if benign is labeled correctly
            # print(ans, self.label)
            self.check_bn = self.label==ans
            # print(self.check_bn)
            self.correct.append(self.check_bn)
            
            # Random agent gonna random
            # obs, info = {}
            # Set state to interceptor
            self.curr = 0
            return self.obs, 0, self.done, {}
                 
    def step_adv(self, action):
        self.iter += 1
        # Remove nan and inf from actions
        action = np.nan_to_num(action, nan=0.0, posinf=2, neginf=-2)
        
        self.converged = self.dist < self.epsilon
        if self.converged or self.iter >= self.steps:
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

        # Setting actions according to vanilla BAGS    
        if self.adaptive == 0 or self.adaptive == 2:
            scale = (1. - max(self.improve_last/50, 1)) + 0.3
            self.action_perlin = 5
            self.action_mask = 1
            self.action_spherical = scale * self.spherical_step
            self.action_source = scale * self.source_step
        
        # generate new advarsarial candidate
        self.candidate = self.generate_boundary_sample(self.wanted_point, self.best_advs, self.x_mask, self.action_source,
                                                     self.action_spherical, self.action_perlin)
        # print(type(self.candidate), type(self.best_advs), type(self.wanted_point))
        self.is_adv, self.logits = self.is_adversarial(ep.astensor(torch.tensor(self.candidate).unsqueeze(0).unsqueeze(1)))
        self.is_adv = self.is_adv.raw.numpy()[0]
        
        # Normal attack flow is interrupted here, generate obs for interceptor
        # wher the final decision on is_adv is made
            
        obs, index, self.lastStep, self.alt = self.observation_int(self.logits, self.candidate)
        r = self.reward_int(self.queues.getStepSizeQueue(index))
        info = self.get_info()
        # Set state to adversary
        self.curr = 1
        return obs, r, self.done, info
        
    def step_ben(self, action):
        self.queries += 1
        candidate, self.label = self.get_benign(action)
        self.logits = self.model(torch.tensor(candidate).unsqueeze(0).unsqueeze(1))
        obs, index, self.lastStep, self.alt = self.observation_int(self.logits, candidate)
        r = self.reward_int(self.queues.getStepSizeQueue(index))
        info = self.get_info()
        # Set state to benign
        self.curr = 2
        return obs, r, self.done, info

    def observation_int(self, logits, query):
        # Intercept the last query, adversarial or benign
        probs = torch.nn.functional.softmax(logits, dim=1)
        # print(type(query))
        index = self.queues.addQuery(torch.tensor(query).unsqueeze(0).unsqueeze(3), probs)
        obs = self.queues.getState(index)
        # print(self.benign)
        # print(bucketIndex)
        lastStep = self.queues.getLastStepQueue(index)
        # print(lastStep)
        origin = self.queues.getOriginQueue(index)

        return obs, index, lastStep, origin
    
    def observation_adv(self):
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
        # observation.extend(pos)
        # observation = np.append(observation, hist)
        # observation.append(self.gain)
        # observation.append(self.iter / self.steps)
        # print(observation)
        
        # Remove nan and inf from observation
        observation = np.nan_to_num(observation, nan=0.0, posinf=1, neginf=0)
        
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

    def reward_int(self, stepsize):
        averageStepsize = np.mean(np.asarray(stepsize))

        if self.past == 1:
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
            # r = abs(math.log(self.gap - l2(self.starting_point, self.best_advs).numpy()) / self.gap) * 0.2
            # Reward staying close to successive best_advs
            r = abs(math.log(self.gap - l2(self.starting_point, self.best_advs) / self.gap)) * 0.2
            # print(self.gain.raw)
            # r = - self.gain
        elif self.past == 2:
            # print(self.check_bn)
            r = 0.5 if self.check_bn else - 0.5
            
        return r

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

    def reward_adv(self, reward_nr):
        if reward_nr == 1:
            # R1
            reward = self.reward1()
        elif reward_nr == 2:
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
    
    def roll_next(self):
        # decide next query; benign with P = ratio_benign
        if random.random() < self.ratio_benign and self.iter > 12:
            self.next = 2
        else:
            self.next = 1
            
    def switch(self, step, action):
        # Return False if candidate lies within the action radius, otherwise True
        # a = torch.tensor(action)
        # b = torch.tensor(step)
        # c = ep.astensor(a < b)
        # If non-adaptive, return actual model decision
        if self.adaptive == 0 or self.adaptive == 1:
            return True
        return action < step

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
    
    def get_benign(self, action):
        nr = random.randint(*self.indices)
        
        mu, sigma = 0, action # mean and standard deviation
        s = np.random.normal(mu, sigma, 28*28)
        # s = torch.tensor(s.reshape(28,28).astype('float32'))
        s = s.reshape(28,28).astype('float32')
        # print(s.shape)
        # print(self.dataset[nr][0].shape)
        s = np.add(self.dataset[nr][0].squeeze(0).numpy(), s)
        benign = np.clip(s,0,1)
        # print(type(benign))
        label = self.dataset[nr][1]
        return benign, label
    
    def get_info(self):
        correct = np.mean(self.correct) if len(self.correct) != 0 else 1
        info = {"iterations" : self.iter,
                "benigns" : self.queries,
                "epsilon" : self.dist,
                "correct" : correct,
                "success" : self.success}
        return info