import logging

import torch.nn as nn


log = logging.getLogger(__name__)


class LeNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LeNet, self).__init__()
        n_channels = input_size[0]
        if tuple(input_size) == (1, 28, 28):
            log.info('Using LeNet architecture for MNIST')
            self.conv1 = nn.Conv2d(n_channels, 20, 5, 1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.fc1   = nn.Linear(50 * 4 * 4, 500)
            self.fc2   = nn.Linear(500, num_classes)
        elif tuple(input_size) == (3, 32, 32):
            log.info('Using LeNet architecture for CIFAR10/100')
            self.conv1 = nn.Conv2d(n_channels, 20, 5, 1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.fc1   = nn.Linear(50 * 5 * 5, 500)
            self.fc2   = nn.Linear(500, num_classes)
        elif tuple(input_size) == (3, 64, 64):
            log.info('Using LeNet architecture for TinyImageNet')
            self.conv1 = nn.Conv2d(n_channels, 20, 5, 1)
            self.pool1 = nn.MaxPool2d(3, 3)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.pool2 = nn.MaxPool2d(3, 3)
            self.fc1   = nn.Linear(50 * 5 * 5, 500)
            self.fc2   = nn.Linear(500, num_classes)
        else:
            raise ValueError(
                'There is no LeNet architecture for an input size {}'.format(input_size))

    def forward(self, x):
        out = nn.functional.relu(self.conv1(x))
        out = self.pool1(out)
        out = nn.functional.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = nn.functional.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def build(input_size, output_size):
    return LeNet(input_size, output_size)


builders = {'lenet': build}
