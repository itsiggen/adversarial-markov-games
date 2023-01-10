import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
import torch.optim as optim
from tqdm import tqdm

class cifarNet(nn.Module):
    def __init__(self):
        super(cifarNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding='same'), nn.PReLU(),
                                  nn.Conv2d(32, 32, 3, padding='same'), nn.PReLU(),
                                  nn.MaxPool2d(2, stride=2))
        
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding='same'), nn.PReLU(),
                                  nn.Conv2d(64, 64, 3, padding='same'), nn.PReLU(),
                                  nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 8 * 8, 512),
                                nn.PReLU(),
                                nn.Linear(512, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256)
                                )
        
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # x = self.dropout(x)
        x = self.conv2(x)
        # print(x.shape)
        # x = self.dropout(x)
        x = x.view(x.size()[0],-1)
        x = self.fc(x)
        return x

    def get_embedding(self, x):
        return self.forward(x)

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)
    
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

# def contrastive_loss(y_true, y_pred):
#     '''Contrastive loss from Hadsell-et-al.'06
#     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     '''
#     margin = 1.0
#     square_pred = K.square(y_pred)
#     margin_square = K.maximum(K.square(margin) - K.square(y_pred), 0)
#     return y_true * square_pred + (1 - y_true) * margin_square


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[]):

    for epoch in tqdm(range(n_epochs), disable=True):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    # print(len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics

def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics

class SiameseCIFAR(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, dataset, mu=0, std=0.1):
        self.cifar_dataset = dataset
        self.train = self.cifar_dataset.train
        # print(self.train)
        self.mu = mu
        self.std = std

        if self.train:
            self.train_labels = self.cifar_dataset.targets
            self.train_data = self.cifar_dataset.data
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(np.asarray(self.train_labels) == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.cifar_dataset.targets
            # print(self.test_labels)
            self.test_data = self.cifar_dataset.data
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(np.asarray(self.test_labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(26)

            positive_pairs = [[i,1,1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i]]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index]
            img1 = transform(img1)
            if target == 1:
                # img2 = img1 + torch.randn(img1.size()) * self.std + self.mu
                img2 = self.apply_transforms(img1, np.random.uniform(-2, 2, 5))
                img2 = torch.clamp(img2, 0, 1)
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
                img2 = self.train_data[siamese_index]
                img2 = transform(img2)
        else:
            target = self.test_pairs[index][2]
            img1 = self.test_data[self.test_pairs[index][0]]
            img1 = transform(img1)
            if target == 0:
                img2 = self.test_data[self.test_pairs[index][1]]
                img2 = transform(img2)
            else:
                # img2 = img1 + torch.randn(img1.size()) * self.std + self.mu
                img2 = self.apply_transforms(img1, np.random.uniform(-2, 2, 5))
                img2 = torch.clamp(img2, 0, 1)
                    
        return (img1, img2), target

    def __len__(self):
        return len(self.cifar_dataset)

    def apply_transforms(self, sample, action):

        sample = sample.transpose(1,2,0)
              
        trs = [transforms.ToTensor()]
        # brightness & contrast
        if action[0] > 0:
            a = action[0]/4
            trs.append(transforms.ColorJitter(brightness=a, contrast=a))
        # rotate
        if action[1] > 0:
            trs.append(transforms.RandomAffine(degrees=action[1]*90))
        # crop
        if action[2] > 0:
            a = action[2]/10
            trs.append(transforms.RandomResizedCrop(size=(32,32), scale=(0.8 - a, 0.8 + a)))
        # translate
        if action[3] > 0:
            a = action[3]/10
            trs.append(transforms.RandomAffine(degrees=0, translate=(a, a)))       

        # compose
        apply = transforms.Compose(trs)
        sample = apply(sample)
                
        # scale
        if action[4] > 0:
            a = action[1]/10
            sc = np.random.uniform(-a, a)
            sample = sample*(1+sc)
            sample = torch.clamp(sample, 0, 1)

        return sample

if __name__ == "__main__":
    
    cuda = torch.cuda.is_available()
    num_classes = 10
    epochs = 10

    transform = transforms.Compose([transforms.ToTensor())

    train_dataset = datasets.CIFAR10('../data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10('../data', train=False, transform=transform, download=True)

    siamese_train_dataset = SiameseCIFAR(train_dataset)
    siamese_test_dataset = SiameseCIFAR(test_dataset)
    batch_size = 256
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
       
    margin = 1.
    embedding_net = cifarNet()
    model = SiameseNet(embedding_net)
    if cuda:
        model.cuda()
    loss_fn = ContrastiveLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 100
    
    # Train and save the model
    fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
    torch.save(model.state_dict(), "CIFARsiamese.pt")
    torch.save(embedding_net.state_dict(), "CIFARembedding.pt")