import eagerpy as ep
import numpy as np
import torch
import torchvision.models as models
from torchvision import datasets, transforms
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import BoundaryAttack, HopSkipJump
from tensorflow.keras.models import Sequential, load_model
from foolbox.criteria import TargetedMisclassification
from models.trainMNISTtorch import Net
import matplotlib.pyplot as plt
import time
import sys
from envs.interceptor import interceptor

# instantiate a MNIST model
transform=transforms.ToTensor()
dataset = datasets.MNIST('./data', train=False,
                        transform=transform, download=True)

model = Net()
model.load_state_dict(torch.load('models/mnist_cnn.pt'))
model.eval()

startImgAttack = dataset[1]
wantedImgAttack = dataset[2]

ara = dataset[0][1]

a = model(startImgAttack[0].unsqueeze(1))
b = ep.argmax(a).detach().numpy()
c = startImgAttack[1]

a == c

# a = interceptor()