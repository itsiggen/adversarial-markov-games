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
from utils.queues import Chain, l2
import utils.pnoise as pn
from utils.utils import get_is_adversarial
from models.trainMNISTtorch import Net
from collections import deque
import math

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
np.seterr(invalid='raise')

class BagsGames(gym.Env):
    def __init__(
        self,
        steps: int = 5000,
        spherical_step: float = 1e-2,
        source_step: float = 1e-2,
        defended = False,
        adaptive: int = 0,
        vanilla = False,
        ratio_benign = 0.5,
        train = True,
        rint = 1,
        radv = 1,
        scale = 20, #default is 20
        dataset = None,
        intercept = 1,
        device = 'cpu',
        tensorboard = False,
        ):
        super(BagsGames, self).__init__()  

        # Boundary Attack inits
        self.steps = steps
        self.spherical_step = spherical_step
        self.source_step = source_step
        self.adaptive = adaptive  # 0: none adaptive | 1: adv adaptive | 2: int adaptive | 3: both adaptive
        self.vanilla = vanilla
        self.ratio_benign = ratio_benign
        self.train = train
        self.rint = rint
        self.radv = radv
        self.intercept = intercept
        self.scale = scale
        self.chain = Chain(nrQueues=3, dataset='mnist')
        
        # random states for benign query and noise generation
        self.rn = np.random.RandomState(1337)
        self.rnn = np.random.RandomState(60)

        # Observation space
        self.observation_spaces = spaces.Dict({
            'adversary': spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32),
            'interceptor': spaces.Box(low=-1, high=1, shape=(77,), dtype=np.float32)
            })

        # Actions space
        self.action_spaces = spaces.Dict({
            'adversary': spaces.Box(low=-2, high=2, shape=(4,), dtype=np.float32),
            'interceptor': spaces.Box(low=-2, high=2, shape=(1,), dtype=np.float32)
            })
        
        # Load MNIST pytorch CNN model -- 99.1% acc -- 98.9% acc adversarially trained
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
    
    def scale_intercept(self, v):
        return ((v + 2) / 4) * self.intercept
    
    def reset(self):
        """ Initialize new targeted attack
        """
        self.iter = 0           # num of attack queries
        self.queries = 0        # num of benign queries
        self.starting_point, startLabel, self.wanted_point, originLabel = self.get_pair()
        # self.starting_point, _ = ep.astensor_(self.starting_point)
        # self.original, self.restore_type = ep.astensor_(self.wanted_point)
        # if self.resets < 3: print("Start:", startLabel, "| Wanted:", originLabel)
        # print("Start:", startLabel, "| Wanted:", originLabel, "\n")
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
        # Reset queues
        self.chain.reset()
        # Target epsilon
        self.epsilon = 1
        self.correct = []
        self.done = False
        self.success = False
        # Set current and next player
        self.curr = 1
        self.next = 0
        self.index = 0

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
        self.stats_is_adv = deque(maxlen=300)
        self.dist_derivative = deque(maxlen=30)
        self.improve_time_avg = deque(maxlen=30)
        self.moving_avg_step_dist = deque(maxlen=30)
        # pos = self.best_advs[::4,::4].flatten()
        
        self.candidate = self.best_advs
    
        # Return observation for interceptor as it's the first agent to move
        # obs, ix, self.lastStep, self.maxStep, self.alt = self.observation_int(self.logits, self.best_advs)
        obs, self.span = self.obs_int(self.logits, self.best_advs)
        
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
            obs, r, done = self.step_int(action)
            self.roll_next()
            info = self.get_info()
            # return obs, r, done, info, self.curr, self.next
            return obs, r, done, info
        elif self.curr == 0:
            if self.next == 1:
                # Adv follows int
                obs, r, done = self.step_adv(action)
            elif self.next == 2:
                # Ben follows int
                obs, r, done = self.step_ben(action)
            self.next = 0
            info = self.get_info()
            return obs, r, done, info
            
    def step_int(self, action):
        # Scale intercept
        action = self.scale_intercept(action)
        if self.curr == 1:
            # Candidate remains adversarial only if outside the containment area
            # print(action, self.lastStep, self.index)
            # candid = self.switch(self.lastStep, self.maxStep, action)
            candid, _ = self.swap(self.span, action)
            # print(self.is_adv, candid)
            self.is_adv = np.logical_and(self.is_adv, candid)
            self.rew_adv = self.is_adv
          
            self.stats_is_adv.append(self.is_adv)
            
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
    
            # store obs so it can be returned from benign
            self.obs = self.obs_adv()

            r = self.reward_adv(self.radv)
            # info = self.get_info()
            # return self.obs, r, self.done
        
        elif self.curr == 2:
            # Classify benign input
            # print(action, self.lastStep)
            # if self.switch(self.lastStep, self.maxStep, action):
            #     # print(np.argsort(torch.nn.functional.softmax(self.logits[0]))[-1])
            #     ans = np.argsort(torch.nn.functional.softmax(self.logits, dim=1))[0][-1]
            # else:
            #     ans = self.alt
            candid, alt = self.swap(self.span, action)
            if candid:
                ans = np.argsort(torch.nn.functional.softmax(self.logits, dim=1))[0][-1]
            else:
                ans = alt
            
            # Check if benign is labeled correctly
            # print(self.label, ans)
            self.check_bn = self.label==ans
            # if self.check_bn:
            #     if not candid:
            #         a = np.argsort(torch.nn.functional.softmax(self.logits, dim=1))[0][-1]
            #         print(self.resets, a, ans, action, self.span)
            # print(self.check_bn)
            self.correct.append(self.check_bn)
            # print(np.mean(self.correct))
            
            # Random agent gonna random
            # obs, info = {}
            r = 0
        
        self.act = action
        self.rad = min(self.span)
        # Set state to interceptor 
        self.curr = 0
        return self.obs, r, self.done
                 
    def step_adv(self, action):
        self.iter += 1
        # Remove nan and inf from actions
        action = np.nan_to_num(action, nan=0.0, posinf=2, neginf=-2)
        
        # self.converged = self.dist < self.epsilon
        # if self.converged or self.iter >= self.steps:
        if self.iter >= self.steps:
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
            scale = (1. - min(self.na_batch/50, 1)) + 0.3
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
        # where the final decision on is_adv is made
            
        # obs, self.index, self.lastStep, self.maxStep, self.alt = self.observation_int(self.logits, self.candidate)
        obs, self.span = self.obs_int(self.logits, self.candidate)
        # if self.index == 1: print(self.index, self.resets)
        # r = self.reward_int(self.queues.getStepSizeQueue(self.index), self.candidate, self.rint)
        r = self.reward_int(self.chain.getStepSizeQueue(0), self.candidate, self.rint)
        # info = self.get_info()
        # Set state to adversary
        self.curr = 1
        return obs, r, self.done
        
    def step_ben(self, action):
        self.queries += 1
        candidate, self.label = self.get_benign(action)
        self.logits = self.model(torch.tensor(candidate).unsqueeze(0).unsqueeze(1))
        # obs, self.index, self.lastStep, self.maxStep, self.alt = self.observation_int(self.logits, candidate)
        obs, self.span = self.obs_int(self.logits, candidate)
        # r = self.reward_int(self.queues.getStepSizeQueue(self.index), candidate, self.rint)
        r = self.reward_int(self.chain.getStepSizeQueue(0), candidate, self.rint)
        # info = self.get_info()
        # Set state to benign
        self.curr = 2
        return obs, r, self.done
    
    # def observation_int(self, logits, query):
    #     # Intercept the last query, adversarial or benign
    #     probs = torch.nn.functional.softmax(logits, dim=1)
    #     # print(type(query))
    #     # index = self.queues.addQuery(torch.tensor(query).unsqueeze(0).unsqueeze(3), probs)
    #     index = self.queues.addQuery(query, probs)
    #     # print(self.next, index+1)
    #     obs = self.queues.getState(index)
    #     # print(obs)
        
    #     # obs = np.random.uniform(0,1,300)
    #     # obs = np.asarray(obs)
    #     # print(obs)
    #     # if self.iter == 2:
    #     #     self.obss = obs
    #     # if self.iter > 2:
    #     #     obs = self.obss
    #     # print(self.benign)
    #     lastStep, maxStep = self.queues.getLastStepQueue(index)
    #     # print(lastStep)
    #     origin = self.queues.getOriginQueue(index)

    #     return obs, index, lastStep, maxStep, origin
    
    def obs_int(self, logits, query):
        # Intercept the last query, adversarial or benign
        probs = torch.nn.functional.softmax(logits, dim=1)
        span = self.chain.checkQuery(query, probs)
        # TODO: try different states
        obs = self.chain.retState()
        obs = np.round(obs,2)
        self.obs_int_state = obs
        # origin = self.chain.getOriginQueue(1)
        # print(lastStep)
        return obs, span
    
    def obs_adv(self):
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
        
        observation.append(self.iter/5000)
        observation.append(np.mean(self.stats_is_adv))
        observation.append(self.gap/15)
        observation.append(self.dist/15)
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

    def reward_int(self, stepsize, cand, reward_nr):
        averageStepsize = np.mean(np.asarray(stepsize))
        #print(averageStepsize)
        r = 0
        if self.past == 1:
            if reward_nr == 1:
                # Reward keeping best_advs close to starting point
                r = abs(math.log((self.gap*0.1 + l2(self.starting_point, self.best_advs)) / self.gap)) * 0.1
            elif reward_nr == 2:
                # reward smalls steps
                r = abs(math.log(averageStepsize,10))
            elif reward_nr == 3:
                # Reward keeping average dist of queries close to starting point
                r = 1 / (l2(self.starting_point, cand) / self.gap)
            elif reward_nr == 4:
                r = -1 if self.rew_adv else 1
            elif reward_nr == 5:
                # reward based on the gap between intercept and smallest span
                r = self.act - min(self.intercept, self.rad)
                
        elif self.past == 2:
            # print(self.check_bn)
            r = 1 if self.check_bn else -1
            # r = min(self.intercept, self.rad) - self.act if self.check_bn else -1
            
        return np.reshape(r, (1,))

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
        reward = 0
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
        # print("rew", reward_nr, reward)

        return np.reshape(reward, (1,))
    
    def roll_next(self):
        # decide next query; benign with P = ratio_benign
        if np.random.random() < self.ratio_benign and self.iter > 5:
            self.next = 2
        else:
            self.next = 1
            
    def switch(self, step, maxstep, action):
        # Return False if candidate lies within the action radius, otherwise True
        # a = torch.tensor(action)
        # b = torch.tensor(step)
        # c = ep.astensor(a < b)
        # If non-adaptive, return actual model decision
        if self.adaptive == 0 or self.adaptive == 1:
            return True
        # if self.iter > 1:
        #     print(step, action, self.curr, maxstep)
            # print(step, action, self.index, self.iter+self.queries, self.resets)
        # if self.curr == 1 and self.iter > 1:
        #     print('0', self.index)
        # if self.curr == 2:
            # print('1', self.index)
        #     print(step, action, "BENIGN")
        return action[0]*maxstep*2 < step
    
    def swap(self, span, action):
        # if self.curr == 2:
        #     print('ADV:', span)
        if action.shape == 1:
            action = action[0]
        if self.adaptive == 0:
            inn = min(span) > 0.05*self.intercept
        elif self.adaptive == 1:
            if self.vanilla:
                inn = min(span) > 0.05*self.intercept
            else:
                inn = True
        else:
            inn = min(span) > action
        # ACHTUNG! before it went
        # alt = self.chain.addQuery(inn)
        if self.train:
            if self.curr == 2:
                alt = self.chain.addQuery(True)
            else:
                alt = self.chain.addQuery(False)
        else:
            alt = self.chain.addQuery(inn)
        return inn, alt

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
        nr = self.rn.randint(*self.indices)
        
        mu, sigma = 0, 0.1 # mean and standard deviation
        s = self.rnn.normal(mu, sigma, self.dim*self.dim)
        # s = torch.tensor(s.reshape(28,28).astype('float32'))
        s = s.reshape(self.dim,self.dim).astype('float32')
        # print(s.shape)
        # print(self.dataset[nr][0].shape)
        s = np.add(self.dataset[nr][0].squeeze(0).numpy(), s)
        benign = np.clip(s,0,1)
        # print(type(benign))
        label = self.dataset[nr][1]
        return benign, label
    
    def get_info(self):
        # correct = np.mean(self.correct) if len(self.correct) != 0 else 1
        correct = np.mean(self.correct) if self.done else 'NA'
        info = {"iterations" : self.iter,
                "benigns" : self.queries,
                "epsilon" : self.dist,
                "correct" : correct,
                "curr" : self.curr,
                "next" : self.next,
                "gap" : self.gap}
        return info