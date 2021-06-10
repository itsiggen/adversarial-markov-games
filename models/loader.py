import numpy as np
from tensorflow.keras.models import Sequential, load_model
from statistics import mean
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, InputLayer
from collections import OrderedDict
import eagerpy as ep
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from models.trainCIFARtorch import resnet20, resnet32
from models.trainMNISTtorch import Net
from models.trainAdvMNISTtorch import LeNet5

def load(dataset, defended):
    if dataset == 'MNIST':
        dataset, model = mnist(defended)
    elif dataset == 'CIFAR':
        dataset, model = cifar(defended)
    return dataset, model

def mnist(defended):
    transform=transforms.ToTensor()
    dataset = datasets.MNIST('../data', train=False, transform=transform, download=True)
    if defended:
        model = LeNet5()
        model.load_state_dict(torch.load('./models/mnist_cnn_adv.pt'))
        model.eval()
    else:
        model = Net()
        model.load_state_dict(torch.load('./models/mnist_cnn.pt'))
        model.eval()
    return dataset, model 

def cifar(defended):
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10('../data', train=False, transform=transform, download=True)
    
    dct = torch.load('../models/CIFARresnet20.th', map_location=torch.device('cpu'))
    state_dict = dct['state_dict']
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model = resnet20()
    model.load_state_dict(new_state_dict)
    model.eval()
        
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return dataset, model