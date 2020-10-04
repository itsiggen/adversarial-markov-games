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

global count
count = 0

class PyTorchAdaptive(PyTorchModel):
    def __init__(self, *args, **kwargs):
        super(PyTorchAdaptive, self).__init__(*args, **kwargs)
        self.calls = 0
    
    def __call__(self, inputs):
        x, restore_type = ep.astensor_(inputs)
        y = self._preprocess(x)
        # plt.figure()
        # plt.imshow(y.squeeze().numpy())
        z = ep.astensor(self._model(y.raw))
        # print(z)
        # print(z.argmax(axis=-1))
        # a = False
        # while not a:
        #     a = check_step()
        #     print('PAASSSING')
        self.calls += 1
        # if self.calls == 100:
        #     sys.exit()
        print(self.calls)
        # print(z.raw)
        return restore_type(z)

def check_step():
    global count
    count += 1
    if count>10:
        return True
    return False
    
# instantiate a MNIST model
transform=transforms.ToTensor()
dataset = datasets.MNIST('./data', train=False,
                       transform=transform, download=True)

model = Net()
model.load_state_dict(torch.load('models/mnist_cnn.pt'))
model.eval()

fmodel = PyTorchAdaptive(model, bounds=(0, 1))
startImgAttack = dataset[1]
wantedImgAttack = dataset[2]
# plt.imshow(startImgAttack[0].squeeze().numpy())
criterion = TargetedMisclassification(torch.tensor([2]))

# a = fmodel(startImgAttack[0].unsqueeze(1))
# print(a.argmax(axis=-1))

# print(wantedImgAttack[0])

# attack = HopSkipJump(steps=64)
attack = BoundaryAttack(steps=50000)
epsilons = [0.3, 1, 3, 10]
advs, _, success = attack(fmodel, wantedImgAttack[0].unsqueeze(1), criterion, epsilons = epsilons, starting_points = startImgAttack[0].unsqueeze(1))
print(success)

plt.imshow(advs[0].squeeze().numpy())

a = fmodel(advs[0])
print(a)