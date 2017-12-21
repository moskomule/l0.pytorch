import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import tensorboardX

from trainer import Trainer
from l0module import L0Linear, L0Conv2d


def get_loader(batch_size):
    data_root = os.path.expanduser('~/.torch/data/mnist')
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_root, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), shuffle=True, batch_size=batch_size)

    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_root, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size=batch_size)
    return train_loader, test_loader


class L0Net(nn.Module):
    """
    based on Caffe LeNet (https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt)
    """

    def __init__(self, mean):
        super(L0Net, self).__init__()
        self.conv1 = L0Conv2d(1, 20, kernel_size=5, stride=1, loc_mean=mean)
        self.conv2 = L0Conv2d(20, 50, kernel_size=5, stride=1, loc_mean=mean)
        self.dense1 = L0Linear(800, 500, loc_mean=mean)
        self.dense2 = L0Linear(500, 10, loc_mean=mean)

    def forward(self, x):
        x, z1 = self.conv1(x)
        x = F.max_pool2d(x, 2, stride=2)
        x, z2 = self.conv2(x)
        x = F.max_pool2d(x, 2, stride=2)
        x = x.view(x.shape[0], -1)
        x, z3 = self.dense1(x)
        x = F.relu(x)
        x, z4 = self.dense2(x)
        penalty = z1 + z2 + z3 + z4
        return x, penalty


class LeNet(nn.Module):
    """
    based on Caffe LeNet (https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt)
    """

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.dense1 = nn.Linear(800, 500)
        self.dense2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, stride=2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, stride=2)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        return x


def main(epochs, batch_size, coef, mean, baseline):
    train_loader, test_loader = get_loader(batch_size)
    logger = tensorboardX.SummaryWriter()

    model = LeNet() if baseline else L0Net(mean)
    model.cuda()

    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    l0_loss = lambda output, target: F.cross_entropy(output[0], target) + coef / len(train_loader.dataset) * output[1]
    loss_f = F.cross_entropy if baseline else l0_loss
    trainer = Trainer(model, optimizer, loss_f, logger)
    if not baseline:
        correct = lambda output, target: (output[0].max(dim=1)[1] == target).sum().data[0]
        trainer.correct = correct

    trainer.start(epochs, train_loader, test_loader)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--coef", type=float, default=1e-1)
    parser.add_argument("--mean", type=float, default=1)
    parser.add_argument("--baseline", action="store_true")

    main(**vars(parser.parse_args()))
