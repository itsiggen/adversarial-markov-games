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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load


# instantiate a MNIST model
# transform=transforms.ToTensor()
# dataset = datasets.MNIST('./data', train=False,
#                         transform=transform, download=True)

# model = Net()
# model.load_state_dict(torch.load('models/mnist_cnn.pt'))
# model.eval()

# startImgAttack = dataset[1]
# wantedImgAttack = dataset[2]

# ara = dataset[0][1]

# a = model(startImgAttack[0].unsqueeze(1))
# b = ep.argmax(a).detach().numpy()
# c = startImgAttack[1]

# a == c

# transform=transforms.ToTensor()
# dataset = datasets.MNIST('./data', train=True,
#                         transform=transform, download=True)

# x_train = dataset.data.view(60000,784)
# y_train = dataset.targets

# # print(x_train.shape)

# # clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
# # print("start training RFC")
# # clf.fit(x_train, y_train)
# # dump(clf, 'RF.joblib')

# clf = load('RF.joblib')

# testset = datasets.MNIST('./data', train=False,
#                        transform=transform)
        
# x_test = testset.data.view(10000,784).detach().numpy()
# y_test = testset.targets.detach().numpy()

# y_pred = clf.predict(x_test)
# acc = accuracy_score(y_test, y_pred)

# class skl_callable():
#     """Wraps a scikit learn model into a callable
#     """
#     def __init__(self, model):
#         self.model = model
    
#     def __call__(self, sample):
#         return self.model.predict_proba(sample.reshape(1, -1))

# b = skl_callable(clf)

# a=0
# for i in x_test:
#     a +=1
#     if a > 10:
#         break
#     print(b(i))

# a = np.linalg.norm([1, 1, 1])


b ='asa'
c = 'dfgdf'

d = b+c

#     else:
#         clf3 = joblib.load('../models/RFC.pkl')
#         predicted = clf3.predict(x_test)
#         print("Accuracy: ", accuracy_score(y_test, predicted))
#         start = time.time()
#         clf3.predict([x_test[1]])
#         end = time.time()
#         print("time needed ",end-start)