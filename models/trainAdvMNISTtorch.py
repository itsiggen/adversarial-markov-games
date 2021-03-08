import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import copy

# Implementation by: https://github.com/wanglouis49/pytorch-adversarial_box

class LinfPGDAttack(object):
    def __init__(self, model=None, epsilon=0.3, k=40, a=0.01, 
        random_start=True):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        if self.rand:
            X = X_nat + np.random.uniform(-self.epsilon, self.epsilon,
                X_nat.shape).astype('float32')
        else:
            X = np.copy(X_nat)

        for i in range(self.k):
            X_var = to_var(torch.from_numpy(X), requires_grad=True)
            y_var = to_var(torch.LongTensor(y))

            scores = self.model(X_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()

            X += self.a * np.sign(grad)

            X = np.clip(X, X_nat - self.epsilon, X_nat + self.epsilon)
            X = np.clip(X, 0, 1) # ensure valid pixel range

        return X
    
def adv_train(X, y, model, criterion, adversary):
    """
    Adversarial training. Returns pertubed mini batch.
    """

    # If adversarial training, need a snapshot of 
    # the model at each batch to compute grad, so 
    # as not to mess up with the optimization step
    model_cp = copy.deepcopy(model)
    for p in model_cp.parameters():
        p.requires_grad = False
    model_cp.eval()
    
    adversary.model = model_cp

    X_adv = adversary.perturb(X.numpy(), y)

    return torch.from_numpy(X_adv)

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def pred_batch(x, model):
    """
    batch prediction helper
    """
    y_pred = np.argmax(model(to_var(x)).data.cpu().numpy(), axis=1)
    return torch.from_numpy(y_pred)


def test(model, loader, blackbox=False, hold_out_size=None):
    """
    Check model accuracy on model based on loader (train or test)
    """
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)

    if blackbox:
        num_samples -= hold_out_size

    for x, y in loader:
        x_var = to_var(x, volatile=True)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data' 
        % (num_correct, num_samples, 100 * acc))

    return acc

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7*7*64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out

# Hyper-parameters
param = {
    'batch_size': 128,
    'test_batch_size': 100,
    'num_epochs': 15,
    'delay': 10,
    'learning_rate': 1e-3,
    'weight_decay': 5e-4,
}


# Data loaders
train_dataset = datasets.MNIST('../data/', train=True, download=True, 
    transform=transforms.ToTensor())
loader_train = torch.utils.data.DataLoader(train_dataset, 
    batch_size=param['batch_size'], shuffle=True)

test_dataset = datasets.MNIST('../data/', train=False, download=True, 
    transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset, 
    batch_size=param['test_batch_size'], shuffle=True)


# Setup the model
net = LeNet5()

if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()
net.train()

# Adversarial training setup
#adversary = FGSMAttack(epsilon=0.3)
adversary = LinfPGDAttack()

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'],
    weight_decay=param['weight_decay'])

for epoch in range(param['num_epochs']):

    print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))

    for t, (x, y) in enumerate(loader_train):

        x_var, y_var = to_var(x), to_var(y.long())
        loss = criterion(net(x_var), y_var)

        # adversarial training
        if epoch+1 > param['delay']:
            # use predicted label to prevent label leaking
            y_pred = pred_batch(x, net)
            x_adv = adv_train(x, y_pred, net, criterion, adversary)
            x_adv_var = to_var(x_adv)
            loss_adv = criterion(net(x_adv_var), y_var)
            loss = (loss + loss_adv) / 2

        if (t + 1) % 100 == 0:
            print('t = %d, loss = %.8f' % (t + 1, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


test(net, loader_test)

torch.save(net.state_dict(), 'mnist_cnn_adv.pt')