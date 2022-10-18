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

# Implenetantion from https://github.com/schoyc/blackbox-detection

def cifar10_encoder(encode_dim=256):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', name='conv2d_1', input_shape=(32, 32, 3)))
    model.add(Activation('relu', name='activation_1'))
    model.add(Conv2D(32, (3, 3), name='conv2d_2'))
    model.add(Activation('relu', name='activation_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1'))
    model.add(Dropout(0.25, name='dropout_1'))

    model.add(Conv2D(64, (3, 3), padding='same', name='conv2d_3'))
    model.add(Activation('relu', name='activation_3'))
    model.add(Conv2D(64, (3, 3), name='conv2d_4'))
    model.add(Activation('relu', name='activation_4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2'))
    model.add(Dropout(0.25, name='dropout_2'))

    model.add(Flatten(name='flatten_1'))
    model.add(Dense(512, name='dense_1'))
    model.add(Activation('relu', name='activation_5'))
    model.add(Dropout(0.5, name='dropout_3'))
    model.add(Dense(encode_dim, name='dense_encode'))
    model.add(Activation('linear', name='encoding'))

    return model

similarityModel = load_model('CIFARencoder.pt', compile=False)
# similarityModel = cifar10_encoder()
# similarityModel.load_weights('CIFARencoder.h5', by_name=True)
# similarityModel.save('CIFARencoder.pt')

# def _weights_init(m):
#     classname = m.__class__.__name__
#     #print(classname)
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#         init.kaiming_normal_(m.weight)

# class LambdaLayer(nn.Module):
#     def __init__(self, lambd):
#         super(LambdaLayer, self).__init__()
#         self.lambd = lambd

#     def forward(self, x):
#         return self.lambd(x)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, option='A'):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != planes:
#             if option == 'A':
#                 """
#                 For CIFAR10 ResNet paper uses option A.
#                 """
#                 self.shortcut = LambdaLayer(lambda x:
#                                             F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
#             elif option == 'B':
#                 self.shortcut = nn.Sequential(
#                      nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                      nn.BatchNorm2d(self.expansion * planes)
#                 )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 16

#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
#         self.linear = nn.Linear(64, num_classes)

#         self.apply(_weights_init)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.avg_pool2d(out, out.size()[3])
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out


# def resnet20():
#     return ResNet(BasicBlock, [3, 3, 3])

# def l2(a,b):
#     #L2 distance
#     return np.linalg.norm(a-b)

# mod = torch.load('CIFARresnet20.th', map_location=torch.device('cpu'))

# state_dict = mod['state_dict']
# # create new OrderedDict that does not contain `module.`
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
# # load params

# model = resnet20()
# model.load_state_dict(new_state_dict)
# model.eval()
# transform = transforms.ToTensor()
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                       std=[0.229, 0.224, 0.225])
# # transform = transforms.Compose([transform, normalize])
# dataset = datasets.CIFAR10('../data', train=False, transform=transform, download=True)
# a = dataset[8][0]
# # print(a)
# b = dataset[8][1]

# a = normalize(a)
# z = ep.argmax(model(a.unsqueeze(0))).detach().numpy()

# s = np.random.normal(0, 1, 32*32*3)
# s = s.reshape(32,32,3).astype('float32')

# g = np.add(a.permute(1,2,0).squeeze(0).numpy(), s)
# g = torch.tensor(g)

# z = ep.argmax(model(g.permute(2,0,1).unsqueeze(0))).detach().numpy()

# similarityModel = load_model('CIFARmodel.h5', compile=False)
# d =  similarityModel.predict(a.permute(1,2,0).unsqueeze(0), steps=1)
# e =  similarityModel.predict(g.unsqueeze(0), steps=1)

# x = l2(d, e)

# print(s)