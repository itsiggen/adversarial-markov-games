from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return output

class LinfPGDAttack(object):
    def __init__(self, model=None, epsilon=0.3, k=40, a=0.01, 
        random_start=True, device='cpu'):
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
        self.device = device
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
            X_var = torch.from_numpy(X).to(self.device)
            X_var.requires_grad=True
            y_var = torch.LongTensor(y).to(self.device)

            # print(next(self.model.parameters()).device)
            scores = self.model(X_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()

            X += self.a * np.sign(grad)

            X = np.clip(X, X_nat - self.epsilon, X_nat + self.epsilon)
            X = np.clip(X, 0, 1) # ensure valid pixel range

        return X
    
def adv_train(X, y, model, adversary):
    """
    Adversarial training. Returns pertubed mini batch.
    """
    # While adversarially training we take a snapshot of 
    # the model at each batch to compute grad and leave 
    # the optimization step unaffected
    model_cp = copy.deepcopy(model)
    for p in model_cp.parameters():
        p.requires_grad = False
    model_cp.eval()
    
    adversary.model = model_cp

    X_adv = adversary.perturb(X.numpy(), y)

    return torch.from_numpy(X_adv)

def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    adversary = LinfPGDAttack(device=device)
    for batch_idx, (data, target) in enumerate(train_loader):
        x, y = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(x)
        # loss = F.nll_loss(output, target)
        loss = criterion(output, y)
        
        if args.adv_train and epoch+1 > args.delay:
            # use predicted label to prevent label leaking
            y_pred = torch.from_numpy(np.argmax(model(x).data.cpu().numpy(), axis=1)) 
            x_adv = adv_train(data, y_pred, model, adversary)
            x_adv = x_adv.to(device)
            # y = y.to(device)
            loss_adv = criterion(model(x_adv), y)
            loss = (loss + loss_adv) / 2
        
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, loader, criterion):
    model.eval()
    test_loss = 0   
    num_correct, num_samples = 0, len(loader.dataset)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            scores = model(x)
            test_loss += criterion(scores, y).item()
            _, preds = scores.data.max(1)
            num_correct += (preds == y).sum()

    test_loss /= len(loader.dataset)
    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data, test loss is: %.4f' 
        % (num_correct, num_samples, 100 * acc, test_loss))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST with optinal adversarial training')
    parser.add_argument('--adv-train', type=bool, default=True,
                        help='adversarial training (default: False)')
    parser.add_argument('--delay', type=int, default=10,
                        help='delay in epochs before adversarial training (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size,
              'pin_memory': False}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )
    
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('../data', train=False,
                       transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset,**kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,**kwargs)

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        scheduler.step()
        criterion = nn.CrossEntropyLoss(reduction='sum')
        test(model, device, test_loader, criterion)
    
    if args.adv_train:
        torch.save(model.state_dict(), "mnist_cnn_adv.pt")
    else:
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()