import eagerpy as ep
import numpy as np
import gym
import torch
import random
import pandas as pd
from foolbox import PyTorchModel
from gym import spaces
from foolbox.criteria import TargetedMisclassification
from utils.utils import flatten, atleast_kd
from utils.queues import Chain, l2, Contrasts
from utils.statefuldefense import StatefulClassifier
import utils.perlin as pn
from data.contrastive_cifar import EmbeddingNet
from models.trainCIFARtorch import resnet20
from torchvision import transforms
from utils.utils import get_is_adversarial
from collections import deque
import math

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
np.seterr(invalid='raise')

class BagsOARSCIFAR(gym.Env):
    def __init__(
        self,
        steps: int = 1000,
        spherical_step: float = 1e-2,
        source_step: float = 1e-2,
        defended = False,
        adaptive: int = 0,
        vanilla = True,
        cont: int = 1,
        ratio_benign = 0.5,
        train = True,
        test = False,
        black = False,
        rint = 1,
        radv = 1,
        step_adapt = 0.667,
        dataset = None,
        intercept = 1,
        ):
        super(BagsOARSCIFAR, self).__init__()  

        # Boundary Attack inits
        self.steps = steps
        self.spherical_step = spherical_step
        self.source_step = source_step
        self.adaptive = adaptive  # 0: none adaptive | 1: adv adaptive | 2: int adaptive | 3: both adaptive
        self.vanilla = vanilla
        self.ratio_benign = ratio_benign
        self.train = train
        self.test = test
        self.black = black
        self.rint = rint
        self.radv = radv
        self.intercept = intercept
        self.step_adapt = step_adapt
        self.chain = Chain(nrQueues=3, simemb=cont, dataset='cifar')
        self.contrasts = Contrasts()
        self.pairs = pd.read_csv('utils/pairs.csv').to_numpy()
        # acts = 9 if self.adaptive == 3 else 4

        # random states for benign query and noise generation
        self.rn = np.random.RandomState(1337)
        self.rnn = np.random.RandomState(60)
        # random state for query draw
        self.rdr = np.random.RandomState(26)

        # Observation space
        self.observation_spaces = spaces.Dict({
            'adversary': spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32),
            'interceptor': spaces.Box(low=-1, high=1, shape=(64,), dtype=np.float32)
            })

        # Actions space
        self.action_spaces = spaces.Dict({
            'adversary': spaces.Box(low=-2, high=2, shape=(4,), dtype=np.float32),
            'interceptor': spaces.Box(low=-2, high=2, shape=(1,), dtype=np.float32)
            })
        
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
        
        # Load contrastive models for state
        self.contrast_model = EmbeddingNet()
        if cont == 1:
            self.contrast_model.load_state_dict(torch.load('models/contrasts/chsja_emb_v.pt', map_location=device))
        elif cont == 2:
            self.contrast_model.load_state_dict(torch.load('models/contrasts/chsja_emb_t.pt', map_location=device))
        elif cont == 0:
            pass
        self.contrast_model.eval()

        # Initialize Blacklight defense
        def_config = {"threshold": 0.5,
                        "add_cache_hit": True,
                        "reset_cache_on_hit": False,
                        "aggregation": "closest",
                        "action": "rejection",
                        "action": "rejection",
                        "state": {
                            "type": "blacklight",
                            "window_size": 20,
                            "num_hashes_keep": 50,
                            "round": 50,
                            "step_size": 1,
                            "num_processes": 5,
                            "input_shape": [3,32,32],
                            "salt": True}}


        self.blacklight = StatefulClassifier(def_config)

        self.model = PyTorchModel(model, bounds=(0, 1), device=device)
        self.indices = [0,7999] if train else [8000,9999]
        self.dim = 32
        self.resets = 0
    
        self.done = False
    
    def scale_intercept(self, v):
        return ((v + 2) / 4) * self.intercept
    
    def reset(self):
        """ Initialize new targeted attack
        """
        self.iter = 0           # num of attack queries
        self.queries = 0        # num of benign queries
        self.det = 0            # num of detected queries
        self.starting_point, startLabel, self.wanted_point, originLabel = self.get_pair()
        self.startLabel = startLabel
        self.originLabel = originLabel
        
        self.resets += 1
        self.criterion = TargetedMisclassification(torch.tensor([startLabel]))
        # Distance between starting and origin point / current best adv
        self.gap = l2(self.starting_point, self.wanted_point)
        # print(self.gap)
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
        # Initialize query queues
        self.chain.reset()
        # Target epsilon
        self.epsilon = 1
        self.correct = []
        self.done = False
        self.success = False
        # Set current and next player
        self.curr = 1
        self.next = 0

        self.scaler = 1

        # Reset Blacklight
        self.blacklight.reset()

        self.is_adversarial = get_is_adversarial(self.criterion, self.model)
        
        if self.starting_point is None:
            raise ValueError("no starting_point provided")
        else:
            self.best_advs = self.starting_point

        cand = self.normalize(torch.tensor(self.best_advs))
        self.is_adv, self.logits  = self.is_adversarial(cand.unsqueeze(0))
        self.is_adv = self.is_adv.cpu().numpy()[0]
        if self.black:
            cache, detected = self.blacklight.forward(cand, self.logits)
            if detected:
                self.det += 1
                self.is_adv = False

        if not self.is_adv:
            raise ValueError("starting_point is not adversarial")

        self.bounds = self.model.bounds
        self.unnormalized_source_direction = self.wanted_point - self.best_advs
        self.source_norm = np.linalg.norm(self.unnormalized_source_direction)
        self.source_direction = self.unnormalized_source_direction / self.source_norm
    
        # create queues to track various statistic used to derive the state
        # success rate, step size, relative location, progress in episode
        self.stats_is_adv = deque(maxlen=20)
        self.avg_adv = 0.5
        self.improve_time_avg = deque(maxlen=30)
        self.moving_avg_step_dist = deque(maxlen=30)
        # pos = self.best_advs[::4,::4].flatten()
        
        self.candidate = self.best_advs
    
        # Return observation for interceptor as it's the first agent to move
        # obs, ix, self.lastStep, self.alt = self.observation_int(self.logits, self.best_advs)
        obs, self.span = self.obs_int(self.logits, self.best_advs, 1)
        
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
            self.decide_next()
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
            if not self.black:
                # Candidate remains adversarial only if outside the containment area
                candid, _ = self.swap(self.span, action)
                self.is_adv = np.logical_and(self.is_adv, candid)
                self.rew_adv = self.is_adv
            self.stats_is_adv.append(self.is_adv)
            self.distance = l2(self.wanted_point, self.candidate)
            self.closer = self.distance < self.source_norm
            is_best_adv = self.is_adv and self.closer
            if is_best_adv:
                self.gain = self.source_norm - self.distance
                self.best_advs = self.candidate
                self.dist = self.distance
                self.gain_moving = self.gain_moving * 0.8 + (self.gain * 0.2) / self.gap
                self.improve_avg = self.improve_avg * 0.8 + (1/(self.improve_last +1))*0.2
                self.reward_mult = self.improve_last
                self.improve_last = 0
                self.na_batch = 1
                self.unnormalized_source_direction = self.wanted_point - self.best_advs
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
    
            # store obs so it can be returned from benign
            self.obs = self.obs_adv()

            r = self.reward_adv(self.radv)
        elif self.curr == 2:
            # Classify benign input
            if self.black:
                if self.miss:
                    ans = np.argsort(torch.nn.functional.softmax(self.logits, dim=1))[0][-2]
                else:
                    ans = np.argsort(torch.nn.functional.softmax(self.logits, dim=1))[0][-1]
            else:
                candid, alt = self.swap(self.span, action)
                if candid:
                    ans = np.argsort(torch.nn.functional.softmax(self.logits, dim=1))[0][-1]
                else:
                    ans = alt
            # Check if benign is labeled correctly
            # print(ans, self.label)
            self.check_bn = self.label==ans
            self.correct.append(self.check_bn)
            r = 0
            
        self.act = action
        self.rad = min(self.span)
        # Set state to interceptor 
        self.curr = 0
        return self.obs, r, self.done
                 
    def step_adv(self, action):
        self.iter += 1
        if self.iter >= self.steps:
            self.done = True
        
        # calculate mask
        mask = np.abs(self.best_advs - self.wanted_point)
        mask /= np.max(mask)
        self.x_mask = mask

        # Setting actions according to vanilla BAGS    
        self.action_perlin = 5
        self.action_mask = 1
        # By default, OARS adapts step size every 20 queries
        if self.iter % 20 == 0:
            # print(self.scaler)
            adv_ratio = np.mean(self.stats_is_adv)
            if adv_ratio < 0.2:
                self.scaler *= self.step_adapt
            elif adv_ratio > 0.5:
                self.scaler /= self.step_adapt

        if self.scaler * self.source_step > 1: self.scaler = 1
        self.action_spherical = self.scaler * self.spherical_step
        self.action_source = self.scaler * self.source_step

        # generate new advarsarial candidate
        self.candidate = self.generate_boundary_sample(self.wanted_point, self.best_advs, self.x_mask, self.action_source,
                                                     self.action_spherical, self.action_perlin)
        cand = torch.tensor(self.candidate)
        cand_normed = self.normalize(cand)
        self.is_adv, self.logits = self.is_adversarial(cand_normed.unsqueeze(0))
        self.is_adv = self.is_adv.numpy()[0]

        if self.black:
            cache, detected = self.blacklight.forward(cand_normed, self.logits)
            if detected:
                self.det += 1
                self.is_adv = False 
            
        # obs, index, self.lastStep, self.alt = self.observation_int(self.logits, self.candidate)
        obs, self.span = self.obs_int(self.logits, cand.numpy(), 1)
        # r = self.reward_int(self.queues.getStepSizeQueue(index), self.candidate, self.rewarder)
        r = self.reward_int(self.chain.getStepSizeQueue(0), self.candidate, self.rint)
        # info = self.get_info()
        # Set state to adversary
        self.curr = 1
        return obs, r, self.done
        
    def step_ben(self, action):
        self.queries += 1
        if self.queries >= self.steps and self.ratio_benign == 1:
            self.done = True
        candidate, self.label = self.get_benign(action)
        self.logits = self.model(self.normalize(torch.tensor(candidate).unsqueeze(0)))
        obs, self.span = self.obs_int(self.logits, candidate, 0)
        # Feed benign to Blacklight
        if self.black:
            cache, self.miss = self.blacklight.forward(self.normalize(torch.tensor(candidate)).to(device), self.logits)

        r = self.reward_int(self.chain.getStepSizeQueue(0), candidate, self.rint)
        # Set state to benign
        self.curr = 2
        return obs, r, self.done
    
    def obs_int(self, logits, query, label):
        # Intercept the last query, adversarial or benign
        probs = torch.nn.functional.softmax(logits, dim=1)
        span = self.chain.checkQuery(query, probs)
        if self.adaptive == 2 or self.adaptive == 3:
            obs, cont = self.chain.getState(2)
            # Add to contrastive dataset
            with torch.no_grad():
                obs = self.contrast_model(cont.view(1,3,25,32,32)).detach().numpy()
        else:
            obs = np.zeros(shape=(64,))
        
        self.obs_int_state = obs
        return obs, span
    
    def obs_adv(self):
        return np.zeros(shape=(8,))
        
    def generate_boundary_sample(self, X_orig, X_adv_current, mask, source_step, spherical_step, perlin_freq):
        # Adapted from FoolBox BoundaryAttack.
        mask = mask ** self.action_mask
        sampling_dir = np.squeeze(pn.create_perlin_noise(self.dim, freq=perlin_freq))

        # calculate candidate on sphere
        dot = np.vdot(sampling_dir, self.source_direction)
        sampling_dir -= dot * self.source_direction  # Project orthogonal to source direction
        sampling_dir *= mask  # Apply regional mask
        sampling_dir /= np.linalg.norm(sampling_dir) + 1e-9  # Norming increases magnitude of masked regions

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
            # r = 1 if self.check_bn else -1
            r = min(self.intercept, self.rad) - self.act if self.check_bn else -1
            
        return np.reshape(r, (1,))

    def reward_adv(self, reward_nr):
        reward = 0
        return np.reshape(reward, (1,))
    
    def decide_next(self):
        # decide next query; benign with P = ratio_benign
        if self.rdr.random() < self.ratio_benign and self.iter > 5:
            self.next = 2
        else:
            self.next = 1
            
    def apply_transforms(self, sample, action):
        trs = []
        # brightness & contrast
        if action[0] > 0:
            a = action[0]/4
            trs.append(transforms.ColorJitter(brightness=a, contrast=a))
        # rotate
        if action[1] > 0:
            trs.append(transforms.RandomAffine(degrees=action[1]*90))
        # crop
        if action[2] > 0:
            a = action[2]/10
            trs.append(transforms.RandomResizedCrop(size=(32,32), scale=(0.8 - a, 0.8 + a)))
        # translate
        if action[3] > 0:
            a = action[3]/10
            trs.append(transforms.RandomAffine(degrees=0, translate=(a, a)))

        # plt.imshow(sample)
        # plt.show()         

        # compose
        apply = transforms.Compose(trs)
        sample = apply(sample)
                
        # scale
        if action[4] > 0:
            a = action[1]/10
            sc = np.random.uniform(-a, a)
            sample = sample*(1+sc)
            sample = torch.clamp(sample, 0, 1)

        # plt.imshow(sample.numpy().transpose(1,2,0))
        # plt.show()
       
        return sample
            
    def swap(self, span, action):
        # if self.curr == 1:
        #     print('ADV:', min(span), action, self.dist, self.avg_adv)
        # if self.curr == 2:
        #     print('BEN:', min(span), action)
        
        if action.shape == 1:
            action = action[0]
        if self.adaptive == 0:
            if self.vanilla:
                inn = min(span) > 0.01*self.intercept
            else:
                inn = True
        elif self.adaptive == 1:
            if self.vanilla:
                inn = min(span) > 0.01*self.intercept
            else:
                inn = True
        else:
            inn = min(span) > action
            # inn = True
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

    def get_benign(self, action):
        nr = self.rn.randint(*self.indices)
        
        mu, sigma = 0, 0.01 # mean and standard deviation
        s = self.rnn.normal(mu, sigma, 3*32*32)
        # s = torch.tensor(s.reshape(28,28).astype('float32'))
        s = s.reshape(3,32,32).astype('float32')
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
                "gap" : self.gap,
                "start": self.startLabel,
                "origin": self.originLabel}
        return info