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
from utils.utils import get_is_adversarial
from typing import List
from models.trainMNISTtorch import Net
from models.trainAdvMNISTtorch import LeNet5
from collections import deque
import matplotlib.pyplot as plt
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.seterr(invalid='raise')

class HsjaGames(gym.Env):
    def __init__(
        self,
        steps: int = 1000,
        reps: int = 64,
        init_gradient_eval_steps: int = 100,
        max_gradient_eval_steps: int = 10000,
        gamma: float = 1.0,
        defended = False,
        adaptive: int = 0,
        ratio_benign = 0.5,
        train = True,
        rewarder = 1,
        dataset = None,
        intercept = 1,
        device = 'cpu',
        tensorboard = False,
        update_stats_every_k: int = 10
        ):
        super(HsjaGames, self).__init__()  

        # Hsja Attack inits
        self.steps = steps
        self.reps = reps
        self.init_grad_evals = init_gradient_eval_steps
        self.max_grad_evals = max_gradient_eval_steps
        self.gamma = gamma
        self.adaptive = adaptive  # 0: none adaptive | 1: adv adaptive | 2: int adaptive | 3: both adaptive
        self.ratio_benign = ratio_benign
        self.rewarder = rewarder
        self.intercept = intercept
        self.tensorboard = tensorboard

        # Observation space
        self.observation_spaces = spaces.Dict({
            'adversary': spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32),
            'interceptor': spaces.Box(low=0, high=1, shape=(300,), dtype=np.float32)
            })

        # Actions space
        self.action_spaces = spaces.Dict({
            'adversary': spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float32),
            'interceptor': spaces.Box(low=-2, high=2, shape=(1,), dtype=np.float32)
            })
        
        # Load MNIST pytorch CNN model -- 99.1% acc -- 98.9% acc adversarially trained
        self.dataset = dataset
        if defended:
            self.mode = LeNet5()
            self.mode.load_state_dict(torch.load('./models/mnist_cnn_adv.pt'))
            self.mode.eval()
        else:
            self.mode = Net()
            self.mode.load_state_dict(torch.load('./models/mnist_cnn.pt'))
            self.mode.eval()

        self.model = PyTorchModel(self.mode, bounds=(0, 1), device=device)
        self.indices = [0,7999] if train else [8000,9999]
        self.dim = 28
        self.resets = 0
    
    def scale_delta(self, v):
        # Delta from [-2,2] to [0.0001,0.0101]
        return ((v + 2) / 400) + 0.0001
    
    def scale_step(self, v):
        # Jump step search from [-2,2] to [0.1,0.9]
        return (v + 2) / 5 + 0.1
    
    # try different num of grad
    def scale_grad(self, v):
        # Gradient estimation steps from [-2,2] to [50,200]
        # return (((v + 2) / 4) * 250 + 50).astype(int)
        return ((v + 2) / 8) + 0.25 # to [0.75,1.25]
    
    def scale_intercept(self, v):
        return ((v + 2) / 4) * self.intercept
    
    def reset(self):
        """ Initialize new HSJA attack
        """
        self.iter = 0           # num of attack queries
        self.reps = 0           # num of attack reps
        self.queries = 0        # num of benign queries
        self.repdone = 0
        self.tb = TensorBoard(logdir=self.tensorboard)
        self.starting_point, startLabel, self.wanted_point, originLabel = self.get_pair()
        # self.starting_point, _ = ep.astensor_(self.starting_point)
        # self.original, self.restore_type = ep.astensor_(self.wanted_point)
        # if self.resets < 5: print("Start:", startLabel, "| Wanted:", originLabel)
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
        # Initialize query queues
        self.queues = Queues(nrQueues=2)
        # Target epsilon
        self.epsilon = 1
        self.correct = []
        self.done = False
        # Set current and next player
        self.curr = 1
        self.next = 0
        self.phase = 2
        self.first_adv = 1

        self.is_adversarial = get_is_adversarial(self.criterion, self.model)
        
        if self.starting_point is None:
            raise ValueError("no starting_point provided")
        else:
            self.best_advs = self.starting_point

        is_adv = self.is_adversarial(ep.astensor(self.best_advs).raw.unsqueeze(1))
        if not is_adv:
            raise ValueError("starting_point is not adversarial")
            
        self.binary_reset(ep.astensor(self.best_advs))
        self.logits, cand = self.binary_query()
      
        # Return observation for interceptor, the first agent to move
        obs, ix, self.lastStep, self.alt = self.observation_int(self.logits, cand.raw)
        
        return obs

    def step(self, action):
        """
        Progress through the internal states of the environment: interceptor 
        always follows after adversary or benign, and adversary, benign or
        interceptor follows after interceptor
        """
        # print(self.iter, self.phase)
        if self.curr == 1 or self.curr == 2:
            # Int responds to adv or ben
            self.past = self.curr
            obs, r, done, info, self.curr, self.next = self.step_int(action)
            # print("int", self.next)
            return obs, r, done, info, self.curr, self.next
        elif self.curr == 0:
            self.past = 0
            if self.next == 1:
                # Adv follows int
                obs, r, done, info = self.step_adv(action)
                self.next = 0
            elif self.next == 2:
                # Ben follows int
                obs, r, done, info = self.step_ben(action)
                self.next = 0
            elif self.next == 0:
                # Int continues responding to adv
                obs, r, done, info, self.curr, self.next = self.step_int(action)    
            # print("nint", self.next)
            return obs, r, done, info, self.curr, self.next
            
    def step_int(self, action):
        # Scale intercept
        action = self.scale_intercept(action)
        if self.curr == 2:
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
        else:
            # Candidate remains adversarial only if outside the containment area
            candid = self.switch(self.lastStep, action)
            # print(self.is_adv, candid)
            self.is_adv = np.logical_and(self.is_adv, candid)
            # proceed according to current phase of HSJA
            self.phase_proceed()
        # roll next query, if int proceed internally to next attack query
        self.next = self.decide_next()
        if self.next == 0:
            self.logits, cand = self.phase_query()
            obs, index, self.lastStep, self.alt = self.observation_int(self.logits, cand.raw)
            r = self.reward_int(self.queues.getStepSizeQueue(index), cand, self.rewarder)
        elif self.next == 1:
            obs = self.observation_adv()
            r = self.reward_adv(self.rewarder)
            # need to return obs and r for both adv and int
        elif self.next == 2:
            # adv shouldn't progress here - return rubbish as state but correct reward
            # self.phase_query()
            # self.obs = self.observation_int()
            # r = self.reward_int(self.rewarder)
            obs = None
            r = None

        info = self.get_info()
        # Set state to interceptor
        self.curr = 0
        return obs, r, self.done, info, self.curr, self.next
        
    def decide_next(self):
        # decide next query; benign with P = ratio_benign
        if self.repdone:
            nxt = 1
        elif np.random.random() < self.ratio_benign and self.iter > 5:
            nxt = 2
        else:
            nxt = 0
        return nxt
      
    def phase_query(self):
        if self.phase == 0: logits, cand = self.grad_query()
        elif self.phase == 1: logits, cand = self.jump_query()
        elif self.phase == 2: logits, cand = self.binary_query()
        return logits, cand

    def phase_proceed(self):
        if self.phase == 0:
            grad, mean = self.grad_proceed()
            if grad is not None:
                self.grad = grad
                self.mean_adv = mean
                self.phase = 1
                self.jump_reset()
        elif self.phase == 1:
            candidate, jsteps = self.jump_proceed()
            if candidate is not None:
                self.candidate = candidate
                # print(jsteps)
                self.phase = 2
                self.binary_reset(ep.astensor(self.candidate))
        elif self.phase == 2:
            adv, bsteps = self.binary_proceed()
            # print(adv)
            if adv is not None:
                self.phase = 0
                self.repdone = 1
                self.best_advs = adv
                self.update()
                 
    def step_adv(self, action):
        action = np.nan_to_num(action, nan=0.0, posinf=2, neginf=-2)
        if self.iter >= self.steps:
            self.tb.close()
            self.done = True
        self.repdone = 0
        self.phase = 0
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
        self.action_binary = 1
        # print(self.action_grad)
        
        # Setting actions according to vanilla HSJA
        if self.adaptive == 0 or self.adaptive == 2:
            self.action_delta = self.select_delta(self.dist)
            # self.action_grad = int(min([self.init_grad_evals * math.sqrt(self.reps), self.max_grad_evals]))
            self.action_grad = num_grad
            self.action_step = 1/math.sqrt(self.reps)
        
        # print(self.action_delta)
        # To force fixed number of queries, reduce gradient estimation steps if necessary
        self.action_grad = min(self.action_grad, max(self.queries_left-16, 16))
       
        self.grad_reset()
        logits, cand = self.grad_query()
        
        # Normal attack flow is interrupted here, generate obs for interceptor
        obs, index, self.lastStep, self.alt = self.observation_int(self.logits, cand.raw)
        r = self.reward_int(self.queues.getStepSizeQueue(index), cand, self.rewarder)
        info = self.get_info()
        # Set state to adversary
        self.curr = 1
        return obs, r, self.done, info
    
    def update(self):
        self.distance = l2(self.wanted_point, self.best_advs.raw.squeeze(0).numpy())
        # Calculate the distance to target gained in the last rep
        self.gain = self.dist - self.distance
        # print('gain:', self.gain)
        self.dist = self.distance
        self.gain_moving = self.gain_moving * 0.2 + (self.gain * 0.8) / self.gap
          
    def step_ben(self, action):
        self.queries += 1
        candidate, self.label = self.get_benign(action)
        self.logits = self.model(torch.tensor(candidate).unsqueeze(0).unsqueeze(1))
        obs, index, self.lastStep, self.alt = self.observation_int(self.logits, torch.tensor(candidate).unsqueeze(0))
        r = self.reward_int(self.queues.getStepSizeQueue(index), candidate, self.rewarder)
        info = self.get_info()
        # Set state to benign
        self.curr = 2
        return obs, r, self.done, info

    def observation_int(self, logits, query):
        # Intercept the last query, adversarial or benign
        probs = torch.nn.functional.softmax(logits, dim=1)
        print(type(query))
        # index = self.queues.addQuery(query.unsqueeze(3), probs)
        index = self.queues.addQuery(query, probs)
        # print(self.next, index+1)
        obs = self.queues.getState(index)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1, neginf=0)
        # print(self.benign)
        # print(bucketIndex)
        lastStep = self.queues.getLastStepQueue(index)
        # if self.iter > 1 and (self.past == 0 or self.past == 1):
        #     if index == 1:
        #         print(lastStep, self.phase, self.next, "adv")
        # print(lastStep)
        origin = self.queues.getOriginQueue(index)

        return obs, index, lastStep, origin
    
    def observation_adv(self):
        # generate observation based on the history of responses
        if self.first_adv:
            observation = []
            # observation.append(np.float32(1.0))
            # TODO: encode source n target class in obs
            observation.append(np.float32(0.5))
            observation.append(np.float32(1.0))
            observation.append(np.float32(1.0))
            observation.append(np.float32(0.5))
            observation.append(np.float32(0.0))
        else:
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
            observation.append(1/self.jsteps)
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
       
   # Decompose steps
   
    def binary_reset(self, best_advs):
        self.bsteps = 0
        # print(len(best_advs))
        # consider using action_binary to go directly to boundary in place of binary search
        # self.action_binary
        self.highs = ep.ones(best_advs, len(best_advs))
        self.lows = ep.zeros_like(self.highs)
        self.threshold = 1 / self.dim ** 3
        self.badvs = best_advs
        self.best_candidate = best_advs
    
    def binary_query(self):
        self.mids = (self.lows + self.highs) / 2
        self.bcand = self.project(self.wanted_point, self.badvs, self.mids)
        self.is_adv, self.logits = self.is_adversarial(ep.astensor(self.bcand).raw.unsqueeze(1))
        self.is_adv = self.is_adv.numpy()[0]
        return self.logits, self.bcand
        
    def binary_proceed(self):
        if self.is_adv:
            self.highs = self.mids
            self.best_candidate = self.bcand
        else:
            self.lows = self.mids
        self.bsteps += 1
        self.iter +=1
        if ep.all(self.highs - self.lows <= self.threshold):
            return self.best_candidate, self.bsteps
        else:
            return None, self.bsteps

    def grad_reset(self):
        self.gsteps = 0
        noise_shape = tuple([self.action_grad] + list(self.best_advs.shape))
        rv = ep.normal(self.best_advs, noise_shape)            
        rv /= atleast_kd(ep.norms.l2(flatten(rv, keep=1), -1), rv.ndim) + 1e-12
        # scaled_rv = atleast_kd(ep.expand_dims(delta, 0), rv.ndim) * rv
        scaled_rv = self.action_delta * rv

        perturbed = ep.expand_dims(self.best_advs, 0) + scaled_rv
        self.perturbed = ep.clip(perturbed, 0, 1)

        self.rv = (self.perturbed - self.best_advs) / 2

        self.multipliers_list: List[ep.Tensor] = []
        
    def grad_query(self):
        # print(self.gsteps)
        self.is_adv, self.logits = self.is_adversarial(ep.astensor(self.perturbed[self.gsteps]).raw.unsqueeze(1))
        self.is_adv = self.is_adv.numpy()[0]
        # print(self.is_adv)
        self.iter += 1
        return self.logits, self.perturbed[self.gsteps]

    def grad_proceed(self):
        self.gsteps += 1
        if self.gsteps < self.action_grad:
            self.multipliers_list.append(ep.ones(self.best_advs,1) if self.is_adv else -ep.ones(self.best_advs,1))
            return None, None
        else:
            # print('done')
            self.multipliers_list.append(ep.ones(self.best_advs,1) if self.is_adv else -ep.ones(self.best_advs,1))
            multipliers = ep.stack(self.multipliers_list, 0)
            # print(multipliers)
            
            vals = ep.where(
                ep.abs(ep.mean(multipliers, axis=0, keepdims=True)) == 1,
                multipliers,
                multipliers - ep.mean(multipliers, axis=0, keepdims=True),
            )
            grad = ep.mean(atleast_kd(vals, self.rv.ndim) * self.rv, axis=0)
    
            grad /= ep.norms.l2(atleast_kd(flatten(grad), grad.ndim)) + 1e-12
            # print('grad steps:', steps)
            return grad, ep.mean(multipliers, axis=0).raw.squeeze(0).numpy()
    
    def jump_reset(self):
        self.jsteps = 0
        self.jeps = self.dist * self.action_step
        # print(grad)

    def jump_query(self):
        self.jcand = ep.clip(self.best_advs + self.jeps * self.grad, 0, 1)
        self.is_adv, self.logits = self.is_adversarial(ep.astensor(self.jcand).raw.unsqueeze(1))
        self.jsteps += 1
        self.iter += 1
        return self.logits, self.jcand
        
    def jump_proceed(self):
        if self.is_adv:
            return self.jcand, self.jsteps
        elif self.jsteps > 9:
            # If jump fails for 10 steps, return last best adv
            return self.best_advs, self.jsteps            
        else:
            self.jeps /= 2
            return None, self.jsteps

    def reward_int(self, stepsize, cand, reward_nr):
        if self.past == 2:
            # print(self.check_bn)
            r = 0.5 if self.check_bn else - 0.5
        if self.past == 0 or self.past == 1:
            # print(stepsize, self.phase)
            averageStepsize = np.mean(np.asarray(stepsize))
            # print(averageStepsize)
            #print(averageStepsize)
            # r = 0 
            if reward_nr == 1:
                # Reward keeping best_advs close to starting point
                r = abs(math.log((self.gap*0.1 + l2(self.starting_point, self.best_advs.raw.squeeze(0).numpy())) / self.gap)) * 0.1
            elif reward_nr == 2:
                # reward small steps
                r = abs(math.log(averageStepsize,10))
            elif reward_nr == 3:
                # Reward keeping average dist of queries close to starting point
                r = 1 / 10*(l2(self.starting_point, cand) / self.gap)
            # r = abs(math.log(abs(averageStepsize/self.step_ref - 1)),10)
            # r = abs(math.log(np.linalg.norm(diff), 10))
            
        return r

    def reward1(self):
        reward = self.gain / self.gap
        return reward

    def reward2(self):
        reward = 1/self.action_grad + 1/self.jsteps + self.reward1()
        return reward

    def reward3(self):
        fraction = self.dist / self.gap
        reward = (1 - fraction ** 2) ** 0.5
        return reward

    def reward4(self):
        reward = 1/self.jsteps + self.reward1()
        return reward
        
    def reward5(self):
        reward = -self.action_grad/1000 + self.reward1()
        return reward

    def reward_adv(self, reward_nr):
        if self.first_adv:
            reward = 0
            self.first_adv = 0
        elif reward_nr == 1:
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

    def project(self, originals, perturbed, epsilons):
        return (1.0 - epsilons) * originals + epsilons * perturbed
    
    def select_delta(self, dist):
        if self.reps == 1:
            result = 0.1 * dist
        else:
            theta = 1 / (self.dim ** 2)
            result = theta * self.dist 
        return result
            
    def switch(self, step, action):
        # Return False if candidate lies within the action radius, otherwise True
        # a = torch.tensor(action)
        # b = torch.tensor(step)
        # c = ep.astensor(a < b)
        # If non-adaptive, return actual model decision
        if self.adaptive == 0 or self.adaptive == 1:
            return True
        # if self.curr == 1 and step > 1:
        #     print(step, action, self.iter)
        # if self.curr == 2:
        #     print(step, action, "BENIGN")
        return action < step
   
    def get_benign(self, action):
        nr = np.random.randint(*self.indices)
        
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
                "gap" : self.gap}
        return info