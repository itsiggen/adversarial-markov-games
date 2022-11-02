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
from utils.queues import Chain, l2, Contrasts
from utils.utils import get_is_adversarial, GameStates
from data.contrastive import EmbeddingNet
from typing import List
from models.trainMNISTtorch import Net
import gc
import math

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
np.seterr(invalid='raise')

class HsjaGames(gym.Env):
    def __init__(
        self,
        steps: int = 5000,
        reps: int = 64,
        init_gradient_eval_steps: int = 100,
        max_gradient_eval_steps: int = 10000,
        gamma: float = 1.0,
        defended = False,
        vanilla = False,
        adaptive: int = 0,
        cont: int = 1,
        ratio_benign = 0.5,
        train = True,
        rint = 1,
        radv = 1,
        scale = 200,
        dataset = None,
        intercept = 1,
        ):
        super(HsjaGames, self).__init__()

        # Hsja Attack inits
        self.steps = steps
        self.reps = reps
        self.init_grad_evals = init_gradient_eval_steps
        self.max_grad_evals = max_gradient_eval_steps
        self.gamma = gamma
        self.adaptive = adaptive  # 0: stateful det | 1: adv adaptive | 2: int adaptive | 3: both adaptive
        self.vanilla = vanilla
        self.ratio_benign = ratio_benign
        self.train = train
        self.rint = rint
        self.radv = radv
        self.scale = scale
        self.intercept = intercept
        self.chain = Chain(nrQueues=3, dataset='mnist')
        
        # for plotting game states
        self.gstates = GameStates()
        self.contrasts = Contrasts()
        # random state for benign query generation
        self.rn = np.random.RandomState(1337)
        # emb = EmbeddingNet()

        # Observation space
        self.observation_spaces = spaces.Dict({
            'adversary': spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32),
            'interceptor': spaces.Box(low=-1, high=1, shape=(64,), dtype=np.float32)
            })

        # Actions space
        self.action_spaces = spaces.Dict({
            'adversary': spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float32),
            'interceptor': spaces.Box(low=-2, high=2, shape=(1,), dtype=np.float32)
            })
        
        # Load MNIST pytorch CNN model -- 99.1% acc -- 98.6% acc adversarially trained
        self.dataset = dataset
        self.mode = Net()
        if defended:
            self.mode.load_state_dict(torch.load('./models/mnist_cnn_adv.pt', map_location=device))
        else:
            self.mode.load_state_dict(torch.load('./models/mnist_cnn.pt', map_location=device))
        self.mode.eval()
            
        # Load contrastive models for state
        self.contrast_model = EmbeddingNet()
        if cont == 1:
            if defended:
                self.contrast_model.load_state_dict(torch.load('models/contrasts/hsja_emb_v.pt', map_location=device))
            else:
                self.contrast_model.load_state_dict(torch.load('models/contrasts/hsja_emb_v.pt', map_location=device))
        elif cont == 2:
            if defended:
                self.contrast_model.load_state_dict(torch.load('models/contrasts/hsja_emb_t.pt', map_location=device))
            else:
                self.contrast_model.load_state_dict(torch.load('models/contrasts/hsja_emb_t.pt', map_location=device))
        elif cont == 0:
            pass
        self.contrast_model.eval()

        self.model = PyTorchModel(self.mode, bounds=(0, 1), device=device)
        self.indices = [0,7999] if train else [8000,9999]
        self.dim = 28
        self.resets = 0
    
    def scale_delta(self, v):
        # Delta from [-2,2] to [0.0001,2.0001]
        base = ((v + 2) / 2) + 0.0001
        return base * self.dist
    
    def scale_step(self, v):
        # Jump step search from [-2,2] to [0.1,1.1]
        return ((v + 2) / 4) + 0.101
    
    # try different num of grad
    def scale_grad(self, v):
        # Gradient estimation steps from [-2,2] to [50,200]
        # return (((v + 2) / 4) * 250 + 50).astype(int)
        return ((v + 2) / 4) + 0.25 # to [0.25,1.25]

    def scale_intercept(self, v):
        return ((v + 2) / 4) * self.intercept
    
    def reset(self):
        gc.collect()
        """ Initialize new HSJA attack
        """
        self.iter = 0           # num of attack queries
        self.reps = 0           # num of attack reps
        self.queries = 0        # num of benign queries
        self.repdone = 0
        self.starting_point, startLabel, self.wanted_point, originLabel = self.get_pair()
        self.startLabel = startLabel
        self.originLabel = originLabel
        # self.starting_point, _ = ep.astensor_(self.starting_point)
        # self.original, self.restore_type = ep.astensor_(self.wanted_point)
        # if self.resets < 5: print("Start:", startLabel, "| Wanted:", originLabel)
        # print("Start:", startLabel, "| Wanted:", originLabel)
        self.resets += 1
        self.criterion = TargetedMisclassification(torch.tensor([startLabel]).to(device))
        # Distance between starting and origin point / current best adv
        self.gap = l2(self.starting_point.cpu(), self.wanted_point.cpu())
        self.dist = self.gap
        self.imp = 0
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
        self.gain = 0
        self.step_moving = 0.1
        self.gain_moving = 0.1
        # Reset queues
        self.chain.reset()
        # Target epsilon
        self.epsilon = 1
        self.correct = []
        self.done = False
        # Set current and next player
        self.past = 1
        self.curr = 1
        self.next = 0
        self.phase = 2
        self.first_adv = 1
        self.collisions = 0

        self.is_adversarial = get_is_adversarial(self.criterion, self.model)
        
        if self.starting_point is None:
            raise ValueError("no starting_point provided")
        else:
            self.best_advs = self.starting_point

        is_adv = self.is_adversarial(ep.astensor(self.best_advs).raw.unsqueeze(1).to(device))
        if not is_adv:
            raise ValueError("starting_point is not adversarial")
            
        self.best_advs = ep.astensor(self.best_advs)
        self.binary_reset(self.best_advs, self.dist)
        self.logits, cand = self.binary_query()
      
        # print(cand.raw.detach().squeeze(0).numpy().shape)
        # Return observation for interceptor, the first agent to move
        # obs, ix, self.lastStep, self.alt = self.observation_int(self.logits, cand.raw.detach().squeeze(0).numpy())
        obs, self.span = self.obs_int(self.logits, cand.raw.detach().squeeze(0).cpu().numpy(), 1)
        
        return obs

    def step(self, action):
        """
        Progress through the internal states of the environment: interceptor 
        always follows after adversary or benign, and adversary, benign or
        interceptor follows after interceptor
        """
        # print(self.iter, self.phase)
        # print(action)
        if self.curr == 1 or self.curr == 2:
            # Int responds to adv or ben
            self.past = self.curr
            obs, r, done, info = self.step_int(action)
            # print("int", self.next)
            # print(info["curr"], info["next"])
            return obs, r, done, info
        elif self.curr == 0:
            if self.next == 1:
                # Adv follows int
                self.next = 0
                obs, r, done, info = self.step_adv(action)
            elif self.next == 2:
                # Ben follows int
                self.next = 0
                obs, r, done, info = self.step_ben(action)
            elif self.next == 0:
                # Int continues responding to adv
                obs, r, done, info = self.step_int(action)
            self.past = 0
            # print("nint", self.next)
            # print(info["curr"], info["next"])
            return obs, r, done, info
            
    def step_int(self, action):
        # Scale intercept
        # print(self.iter)
        action = self.scale_intercept(action)
        if self.curr == 2:
            # Classify benign input
            # if self.switch(self.lastStep, action):
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
            # print(ans, self.label)
            self.check_bn = self.label==ans
            # print(self.check_bn)
            self.correct.append(self.check_bn)
        else:
            # Candidate remains adversarial only if outside the containment area
            # candid = self.switch(self.lastStep, action)
            candid, _ = self.swap(self.span, action)
            # Is_det: if adversarial & inside containment
            self.is_det = np.logical_and(self.is_adv, not candid)
            # Is adv only if it actually is AND is out of containment area
            self.is_adv = np.logical_and(self.is_adv, candid)
            self.rew_adv = self.is_adv
            # proceed according to current phase of HSJA
            self.phase_proceed()
        # roll next query, if int proceed internally to next attack query
        self.next = self.decide_next()
        self.act = action
        self.rad = min(self.span)
        if self.next == 0:
            self.logits, cand = self.phase_query()
            # obs, index, self.lastStep, self.alt = self.observation_int(self.logits, cand)
            obs, self.span = self.obs_int(self.logits, cand, 1)
            # r = self.reward_int(self.queues.getStepSizeQueue(index), cand, self.rint)
            r = self.reward_int(self.chain.getStepSizeQueue(0), cand, self.rint)
        elif self.next == 1:
            obs = self.obs_adv()
            r = self.reward_adv(self.radv)
            # need to return obs and r for both adv and int
        elif self.next == 2:
            # adv shouldn't progress here - return rubbish as state but correct reward
            # self.phase_query()
            # self.obs = self.observation_int()
            # r = self.reward_int(self.rewarder)
            obs = None
            r = None

        # Set state to interceptor
        self.curr = 0
        self.imp = 0
        info = self.get_info()
        return obs, r, self.done, info
        
    def decide_next(self):
        # decide next query; benign with P = ratio_benign
        if self.repdone:
            nxt = 1
        elif np.random.random() < self.ratio_benign and self.iter > 5:
        # elif self.iter % 3 == 0 and self.past !=2 and self.iter > 5:
            # print(self.iter)
            nxt = 2
        else:
            nxt = 0
        return nxt
      
    def phase_query(self):
        if self.phase == 0: logits, cand = self.grad_query()
        elif self.phase == 1: logits, cand = self.jump_query()
        elif self.phase == 2: logits, cand = self.binary_query()
        return logits.cpu(), cand.raw.detach().squeeze(0).cpu().numpy()

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
                self.binary_reset(ep.astensor(self.candidate), self.dist)
        elif self.phase == 2:
            adv, bsteps = self.binary_proceed()
            # print(adv)
            if adv is not None:
                self.phase = 0
                self.repdone = 1
                if l2(self.wanted_point.cpu(), adv.raw.squeeze(0).cpu().numpy()) < self.dist:
                    self.best_advs = adv
                self.update()
                 
    def step_adv(self, action):
        action = np.nan_to_num(action, nan=0.0, posinf=2, neginf=-2)
        if self.iter >= self.steps:
            self.done = True
        self.repdone = 0
        self.phase = 0
        self.reps += 1
        self.queries_left = self.steps - self.iter 
                    
        # Scale actions to proper values
        # self.action_delta = 0.1*self.dist if self.reps == 1 else self.scale_delta(action[0])*self.dist
    
        self.action_delta = self.scale_delta(action[0])
        # if self.reps == 1: self.action_delta = 0.1*self.dist
        # self.action_step = 1.0 if self.reps == 1 else self.scale_step(action[1])
        self.action_step = self.scale_step(action[1])
        self.action_grad = self.scale_grad(action[2])
        num_grad = int(min([self.init_grad_evals * math.sqrt(self.reps), self.max_grad_evals]))
        self.action_grad = (num_grad * self.action_grad).astype(int)
        # self.action_binary = self.scale_binary(action[3])
        # print(self.action_grad)
        
        # Setting actions according to vanilla HSJA
        if self.adaptive == 0 or self.adaptive == 2:
            self.action_delta = self.select_delta(self.dist)
            # self.action_grad = int(min([self.init_grad_evals * math.sqrt(self.reps), self.max_grad_evals]))
            # self.action_grad = int(num_grad/3) if self.train else num_grad
            self.action_grad = num_grad
            # self.action_grad = int(num_grad/10)
            # self.action_grad = num_grad
            self.action_step = 1/math.sqrt(self.reps)
        
        # print(self.action_delta)
        # To force fixed number of queries, reduce gradient estimation steps if necessary
        self.action_grad = min(self.action_grad, max(self.queries_left-16, 16))
       
        self.grad_reset()
        logits, cand = self.grad_query()
        
        # Normal attack flow is interrupted here, generate obs for interceptor
        # obs, index, self.lastStep, self.alt = self.observation_int(self.logits, cand.raw.detach().squeeze(0).numpy())
        obs, self.span = self.obs_int(self.logits, cand.raw.detach().squeeze(0).cpu().numpy(), 1)
        # r = self.reward_int(self.queues.getStepSizeQueue(index), cand, self.rint)
        r = self.reward_int(self.chain.getStepSizeQueue(0), cand, self.rint)
        # Set state to adversary
        self.curr = 1
        info = self.get_info()
        return obs, r, self.done, info
    
    def update(self):
        distance = l2(self.wanted_point.cpu(), self.best_advs.raw.squeeze(0).cpu().numpy())
        # print(self.gap, distance)
        # Calculate the distance to target gained in the last rep
        self.gain = self.dist - distance
        # print('gain:', self.gain)
        self.dist = distance
        self.gain_moving = self.gain_moving * 0.2 + (self.gain * 0.8) / self.gap
          
    def step_ben(self, action):
        self.queries += 1
        candidate, self.label = self.get_benign(action)
        self.logits = self.model(torch.tensor(candidate).unsqueeze(0).unsqueeze(1).to(device))
        self.logits = self.logits.cpu()
        # obs, index, self.lastStep, self.alt = self.observation_int(self.logits, candidate)
        obs, self.span = self.obs_int(self.logits, candidate, 0)
        
        # if (np.asarray(obs)[0:6] < 0.3).any():
        #     self.collisions += 1
        # print(self.collisions/self.queries)
        # print("DEBN")
        # r = self.reward_int(self.queues.getStepSizeQueue(index), candidate, self.rint)
        r = self.reward_int(self.chain.getStepSizeQueue(0), candidate, self.rint)
        # Set state to benign
        self.curr = 2
        info = self.get_info()
        return obs, r, self.done, info

    # def observation_int(self, logits, query):
    #     # Intercept the last query, adversarial or benign
    #     probs = torch.nn.functional.softmax(logits, dim=1)
    #     # print(type(query))
    #     # index = self.queues.addQuery(query.unsqueeze(3), probs)
    #     index = self.queues.addQuery(query, probs)
    #     # print(self.next, index+1)
    #     obs = self.queues.getState(index)
    #     obs = np.nan_to_num(obs, nan=0.0, posinf=1, neginf=0)
    #     # print(self.benign)
    #     # print(bucketIndex)
    #     lastStep = self.queues.getLastStepQueue(index)
    #     # if self.iter > 1 and (self.past == 0 or self.past == 1):
    #     #     if index == 1:
    #     #         print(lastStep, self.phase, self.next, "adv")
    #     # print(lastStep)
    #     origin = self.queues.getOriginQueue(index)

    #     return obs, index, lastStep, origin
    
    def obs_int(self, logits, query, label):
        # Intercept the last query, adversarial or benign
        probs = torch.nn.functional.softmax(logits, dim=1)
        # print(np.around(np.asarray(probs), decimals=4))
        span = self.chain.checkQuery(query, probs)
        # trivial state for non-adaptive
        if self.adaptive == 2 or self.adaptive == 3:
            obs, cont = self.chain.getState(2)
            # Add to contrastive dataset
            # self.contrasts.add(cont, label)
            # print(cont.shape)
            with torch.no_grad():
                obs = self.contrast_model(torch.unsqueeze(cont, dim=0)).detach().numpy()
        else:
            obs = np.zeros(shape=(64,))
        # print(max(max(obs)))
        # print('CURRENT:', self.curr)
        # obs = np.round(obs,2)
        self.obs = obs
        # if self.next == 0 or self.next == 1:
        # print(obs)
        # print("STATE", obs[0:6], obs[25:31], obs[50:60])
        # origin = self.chain.getOriginQueue(1)
        # print(lastStep)
        return obs, span
    
    def obs_adv(self):
        # generate observation based on the history of responses
        if self.first_adv:
            observation = []
            # observation.append(np.float32(1.0))
            # TODO: encode source n target class in obs
            observation.append(np.float32(0.0))
            observation.append(np.float32(0.5))
            observation.append(np.float32(1.0))
            observation.append(self.gap/15)
            observation.append(self.dist/15)
            observation.append(np.float32(1.0))
            observation.append(np.float32(0.5))
            observation.append(np.float32(0.0))
        else:
            # Use dist in place of moving dist
            loc = self.dist / self.gap
            self.dist_moving = self.dist_moving * 0.2 + (loc) * 0.8
            slope = self.dist_moving - loc
            # Observation should also reflect the trajectory taken
            # by the binary search, grad approximation, and jump step
            observation = []
            # observation.append(1/self.bin_steps)
            observation.append(self.iter/5000)
            observation.append((self.mean_adv+1)/2)
            observation.append(1/self.jsteps)
            observation.append(self.gap/15)
            observation.append(self.dist/15)
            observation.append(loc)
            observation.append(slope)
            observation.append(self.gain/self.gap)
            
            observation = np.nan_to_num(observation, nan=0.0, posinf=1, neginf=0)
            
        return observation
       
   # Decompose steps
   
    def binary_reset(self, best_advs, dist):
        self.bsteps = 0
        # print(len(best_advs))
        self.highs = ep.ones(best_advs, len(best_advs))
        self.lows = ep.zeros_like(self.highs)
        self.threshold = 1 / self.dim ** 3
        self.badvs = best_advs
        self.best_candidate = best_advs
        self.bloc = dist
        # self.bcand = self.best_advs
    
    def binary_query(self):
        self.mids = (self.lows + self.highs) / 2
        # self.pcand = self.bcand
        self.bcand = self.project(self.wanted_point, self.badvs, self.mids)
        # print(l2(self.pcand.raw.squeeze(0).numpy(), self.bcand.raw.squeeze(0).numpy()))
        # fig = plt.figure
        # plt.imshow(self.bcand.raw.squeeze(0).numpy(), cmap='gray')
        # plt.show()
        self.is_adv, self.logits = self.is_adversarial(ep.astensor(self.bcand).raw.unsqueeze(1).to(device))
        self.is_adv = self.is_adv.cpu().numpy()[0]
        self.logits = self.logits.cpu()
        self.iter += 1
        # print(self.bsteps, self.bcand.sum(), self.starting_point.sum())
        return self.logits, self.bcand
        
    def binary_proceed(self):
        if self.is_adv:
            # print("IS")
            self.highs = self.mids
            self.best_candidate = self.bcand
            loc = l2(self.wanted_point.cpu(), self.bcand.raw.squeeze(0).cpu().numpy())
            if loc < self.bloc:
                self.imp = 1
                self.bloc = loc
            # self.best_advs = self.bcand
        else:
            self.lows = self.mids
            self.imp = 0
        self.bsteps += 1
        if self.highs - self.lows <= self.threshold:
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
        self.is_adv, self.logits = self.is_adversarial(ep.astensor(self.perturbed[self.gsteps]).raw.unsqueeze(1).to(device))
        self.is_adv = self.is_adv.cpu().numpy()[0]
        self.logits = self.logits.cpu()
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
            return grad, ep.mean(multipliers, axis=0).raw.squeeze(0).cpu().numpy()
    
    def jump_reset(self):
        self.jsteps = 0
        self.jeps = self.dist * self.action_step
        # print(grad)

    def jump_query(self):
        self.jcand = ep.clip(self.best_advs + self.jeps * self.grad, 0, 1)
        self.is_adv, self.logits = self.is_adversarial(ep.astensor(self.jcand).raw.unsqueeze(1).to(device))
        self.is_adv = self.is_adv.cpu().numpy()[0]
        self.logits = self.logits.cpu()
        self.jsteps += 1
        self.iter += 1
        return self.logits, self.jcand
        
    def jump_proceed(self):
        # print(l2(self.best_advs.raw.squeeze(0).numpy(), self.jcand.raw.squeeze(0).numpy()), self.jsteps)
        if self.is_adv:
            return self.jcand, self.jsteps
        elif self.jsteps > 9:
            # If jump fails for 10 steps, return last best adv
            return self.best_advs, self.jsteps            
        else:
            self.jeps /= 2
            return None, self.jsteps

    def reward_int(self, stepsize, cand, reward_nr):
        r = 0
        if self.past == 0 or self.past == 1:
            # averageStepsize = np.mean(np.asarray(stepsize))
            # print(averageStepsize)
            if reward_nr == 1:
                # Reward keeping best_advs close to starting point
                # r = max(-math.log((self.gap*0.1 + l2(self.starting_point, self.best_advs)) / self.gap), 0)
                # r = abs(math.log((self.gap*0.1 + l2(self.starting_point.detach().numpy(), self.best_advs.raw.squeeze(0).numpy())) / self.gap))
                r = 1 - 2*(l2(self.starting_point.detach().numpy(), self.best_advs.raw.squeeze(0).numpy()) / self.gap)
            elif reward_nr == 2:
                # reward interception + penalty/bonus on binary queries
                r = self.act - min(self.intercept, self.rad)
                if self.imp:
                    r -= 2
                else:
                    r += 2
            elif reward_nr == 3:
                # Reward keeping average dist of queries close to starting point
                # r = 1 / 10*(l2(self.starting_point.detach().numpy(), cand) / self.gap)
                # r = max(r,1)
                # print(self.imp)
                r = -2 if self.imp else 0
            elif reward_nr == 4:
                # reward small gains
                # r = 1 / 10*self.gain
                # reward interception
                # print(self.rew_adv)
                r = -1 if self.rew_adv else 1
            elif reward_nr == 5:
                # reward based on the gap between intercept and smallest span
                r = self.act - min(self.intercept, self.rad)
                # print(r, 'sad')
            # elif reward_nr == 6:
            #     # reward intercepting jump and binary steps
            #     r = 0
            #     if self.phase == 1 or self.phase == 2:
            #         r = self.act - min(self.intercept, self.rad)
            elif reward_nr == 6:
                # reward intercepting jump and binary steps
                r = 0
                if self.phase == 1 or self.phase == 2:
                    r = np.sign(self.act - min(self.intercept, self.rad))*1
            elif reward_nr == 7:
                # reward when intercepting queries that are actually adversarial
                r = 0
                if self.is_det: r = 1
            elif reward_nr == 8:
                # reward when intercepting queries that are actually adversarial proportional to the gap
                r = 0
                if self.is_det:
                    r = self.act - self.rad
                
        elif self.past == 2:
            # print(self.check_bn)
            # r = 0.5 if self.check_bn else -0.5
            r = min(self.intercept, self.rad) - self.act if self.check_bn else -1
            # r = 0.5 if self.check_bn else -0.5
            # print(r, 'jkh')
            
        return np.reshape(r, (1,))

    def reward1(self):
        # reward = self.gain / self.gap
        reward = 2*self.gain
        return reward
    
    def reward2(self):
        reward = -self.action_grad/1000 + self.reward1()
        return reward

    def reward3(self):
        reward = 10*self.gain / self.dist
        # reward = self.gain
        return reward

    def reward4(self):
        reward = 5/self.dist
        return reward

    def reward5(self):
        reward = 0
        if self.iter >= self.steps:
            reward = 2*(self.gap - self.dist)/self.gap
        return reward

    def reward_adv(self, reward_nr):
        reward = 0
        if self.first_adv:
            # reward = 0
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

        return np.reshape(reward, (1,))

    def get_pair(self):
        startImgNr = random.randint(*self.indices)
        originImgNr = random.randint(*self.indices)
        
        # Make sure original image is correctly classified by the model
        while not ep.argmax(self.model(self.dataset[originImgNr][0].to(device).unsqueeze(1))).cpu().detach().numpy() == self.dataset[originImgNr][1]:
            originImgNr = random.randint(*self.indices)
        
        # Make sure starting and original images do not belong to the same class, and starting is correctly classified
        while self.dataset[startImgNr][1] == self.dataset[originImgNr][1] \
            or not ep.argmax(self.model(self.dataset[startImgNr][0].to(device).unsqueeze(1))).cpu().detach().numpy() == self.dataset[startImgNr][1]:
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
    
    def swap(self, span, action):
        if action.shape == 1:
            action = action[0]
        # if self.curr == 0 or self.curr == 1:
        #     if self.phase == 1 or self.phase == 2:
        #         print('ADV:', span, self.curr)
        #     self.gstates.add(1, self.obs, self.phase, min(span), action)
            # if self.phase == 1 or self.phase == 2:
                # print("STATEA", self.phase, self.iter, '%.2f'%sum(self.obs[0:25]), '%.2f'%sum(self.obs[25:50]), '%.2f'%sum(self.obs[50:75]), '%.2f'%sum(self.obs[75:100]), '%.2f'%self.obs[100], '%.2f'%self.obs[101])
            # if self.phase == 2:
            # print("STATE1", self.phase, '%.2f'%sum(self.obs[0:25]), '%.2f'%sum(self.obs[25:50]), '%.2f'%sum(self.obs[50:75]), '%.2f'%self.obs[75], '%.2f'%self.obs[76])
            # print(min(span), action[0], self.dist, self.phase, self.iter)
            # print(self.phase, self.iter)
        # if self.curr == 2:
        #     print('BEN:', span, self.curr)
        #     self.gstates.add(0, self.obs, 3, min(span), action)
            # print("STATEB", '%.2f'%sum(self.obs[0:25]), '%.2f'%sum(self.obs[25:50]), '%.2f'%sum(self.obs[50:75]), '%.2f'%sum(self.obs[75:100]), '%.2f'%self.obs[100], '%.2f'%self.obs[101])
            # print(min(span), action[0], 'ben')
            # if min(span) < action[0]:
        #         print("STATE", self.obs[0:6], self.obs[25:31], self.obs[50:60])
        #         print(min(span), action[0], self.phase, self.queries, '-------------------------------------------------------------------BENIGN')
            # else:
            #     print(min(span), action[0], self.phase, self.queries, 'ben')
        #     if min(span) < action[0]: print("\n") 
            # print('benign')
        # if self.phase == 2:
        #     print(min(span), self.bsteps)
        if self.adaptive == 0:
            inn = min(span) > 0.05*self.intercept
        elif self.adaptive == 1:
            if self.vanilla:
                inn = min(span) > 0.05*self.intercept
            else:
                inn = True
        else:
            inn = min(span) > action
            # inn = min(span) > 0.9
        # inn = False
        # train: adv and ben go to correct queue / test: the agent decides
        if self.train:
            if self.curr == 2:
                alt = self.chain.addQuery(True)
            else:
                alt = self.chain.addQuery(False)
        else:
            alt = self.chain.addQuery(inn)
            # if self.curr != 2 and inn:
            #     print(self.iter)
        # if self.curr == 0 or self.curr == 1:    
        #     print(inn, self.is_adv)
        return inn, alt
   
    def get_benign(self, action):
        nr = self.rn.randint(*self.indices)

        # mu, sigma = 0, action # mean and standard deviation
        mu, sigma = 0, 0.1 # mean and standard deviation
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
    
    # def get_benign(self, action):
    #     while True:
    #         nr = np.random.randint(*self.indices)
    #         label = self.dataset[nr][1]
    #         if label == self.startLabel: break
    #         # if self.train or self.iter % 2 == 0 or label == self.startLabel: break
    #     # mu, sigma = 0, action # mean and standard deviation
    #     mu, sigma = 0, 0.1 # mean and standard deviation
    #     s = np.random.normal(mu, sigma, 28*28)
    #     # s = torch.tensor(s.reshape(28,28).astype('float32'))
    #     s = s.reshape(28,28).astype('float32')
    #     # print(s.shape)
    #     # print(self.dataset[nr][0].shape)
    #     s = np.add(self.dataset[nr][0].squeeze(0).numpy(), s)
    #     benign = np.clip(s,0,1)
    #     # print(type(benign))
    #     return benign, label
    
    # activation = {}
    # def get_activation(name):
    #     def hook(model, input, output):
    #         activation[name] = output.detach()
    #     return hook
    
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