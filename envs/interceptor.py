import warnings
import random
import gym
from gym import spaces
import eagerpy as ep
import numpy as np
import torch
from torchvision import datasets, transforms
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import BoundaryAttack, HopSkipJump
from foolbox.criteria import TargetedMisclassification
from utils.utils import MisdirectedMisclassification, l2
from models.trainMNISTtorch import Net
from syncopation import PyTorchAdaptive
import matplotlib.pyplot as plt


class interceptor(gym.Env):
    """Custom Environment that follows gym interface
    Represents adaptive control in defensive policies
    """
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super().__init__()
        
        self.interceptors = 1
        
        # Actions controlled by the interceptor
        self.action_space_int = spaces.Discrete(3)
        # Observation space is the MNIST inputs
        self.observation_space_int = spaces.Box(low=0, high=1, shape=(28, 28), dtype=np.float32)

        # Load MNIST model
        transform=transforms.ToTensor()
        self.dataset = datasets.MNIST('../data', train=False, transform=transform, download=True)
        self.model = Net()
        self.model.load_state_dict(torch.load('models/mnist_cnn.pt'))
        self.model.eval()
        
        # Create Foolbox model 
        self.fmodel = PyTorchAdaptive(self.model, bounds=(0, 1))
        
        self.ready = False
        self.reward = 0
        self.done = False

    def observe(self, agent):
        board_vals = np.array(self.board.squares).reshape(3, 3)
        cur_player = self.agents.index(self.agent_selection)
        opp_player = (cur_player + 1) % 2

        cur_p_board = np.equal(board_vals, cur_player + 1)
        opp_p_board = np.equal(board_vals, opp_player + 1)
        return np.stack([cur_p_board, opp_p_board], axis=2).astype(np.int8)
    
    def reset(self):
        """At reset, initialize a new targeted attack
        """
        startImg, originImg, originLabel = self.get_pair()
        criterion = TargetedMisclassification(torch.tensor([originLabel]))
        misterion = MisdirectedMisclassification(torch.tensor([originLabel]))

        self.attack = BoundaryAttack(steps=20000)
        epsilon = 3
        advs, _, success = self.attack(self.fmodel,
                                       wantedImgAttack[0].unsqueeze(1),
                                       criterion,
                                       misterion,
                                       epsilons = epsilon,
                                       starting_points = startImgAttack[0].unsqueeze(1)
                                       )
        
       
        
        a = self.fmodel(advs[0])
        print(a)


        self.rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}

        # selects the first agent
        self.agent_selection = self._agent_selector.reset()


    # Action is a value from 0 to 2, indicating the interceptor response
    def step(self, action):
        # Perform next step in the boundary attack
        done, response = self.attack.step(action)
        if done:
            adv, dst, is_adv = self.check_success()
        # check if we exceeded query budget
        done = bool(self.query_current >= self.query_max)
        # done = bool(self.query_current >= self.query_max or self.dist_opt_adv<3)
        self.query_current += 1
        observation = self.gen_observation()

        reward = self.reward(1)

        # Return current agent
        info = self.agent_selection

        return observation, reward, done, info

    def get_pair(self):
        startImgNr = random.randint(0,10000)
        originImgNr = random.randint(0,10000)
        
        # Make sure original image is correctly classified by the model
        while not ep.argmax(self.model(self.dataset[originImgNr][0].unsqueeze(1))).detach().numpy() == self.dataset[originImgNr][1]:
            originImgNr = random.randint(0,1000)
        
        # Make sure starting and original images do not belong to the same class, and starting is correctly classified
        while self.dataset[startImgNr][1] == self.dataset[originImgNr][1] \
            or not ep.argmax(self.model(self.dataset[startImgNr][0].unsqueeze(1))).detach().numpy() == self.dataset[startImgNr][1]:
            startImgNr = random.randint(0,1000)
        
        startImg = self.dataset[startImgNr][0]
        
        originImg = self.dataset[originImgNr][0]
        originLabel = self.dataset[originImgNr][1]
        
        return startImg, originImg, originLabel
    
    def check_success(self, original, best_adv, epsilon):
        # Clip perturbations by epsilons and validate which are still adversarial

        xpc = l2.clip_perturbation(original, best_adv, epsilon)
        dst = l2(original, best_adv)
        is_adv = self.attack.is_adversarial(xpc)
        
        return xpc, dst, is_adv
        
    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # Print relevant metrics at fixed intervals
        if self.query_current % 99 == 0:
            # source step - spherical step
            print(f"current step : {self.query_current}")
            print(f"distant adv : {self.dist_opt_adv}")
            
        
    def close(self):
        pass