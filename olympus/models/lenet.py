import logging
import numpy
import torch.nn as nn

from olympus.utils import info


class LeNet(nn.Module):
    """
    `Paper <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.

    Attributes
    ----------
    input_size: (1, 28, 28), (3, 32, 32), (3, 64, 64)
        Supported input sizes

    References
    ----------
    .. [1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner.
        "Gradient-based learning applied to document recognition."
        Proceedings of the IEEE, 86(11):2278-2324, November 1998.
    """
    def __init__(self, input_size, num_classes):
        super(LeNet, self).__init__()

        if not isinstance(num_classes, int):
            num_classes = numpy.product(num_classes)

        n_channels = input_size[0]
        if tuple(input_size) == (1, 28, 28):
            info('Using LeNet architecture for MNIST')
            self.conv1 = nn.Conv2d(n_channels, 20, 5, 1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.fc1   = nn.Linear(50 * 4 * 4, 500)
            self.fc2   = nn.Linear(500, num_classes)
        elif tuple(input_size) == (3, 32, 32):
            info('Using LeNet architecture for CIFAR10/100')
            self.conv1 = nn.Conv2d(n_channels, 20, 5, 1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.fc1   = nn.Linear(50 * 5 * 5, 500)
            self.fc2   = nn.Linear(500, num_classes)
        elif tuple(input_size) == (3, 64, 64):
            info('Using LeNet architecture for TinyImageNet')
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
