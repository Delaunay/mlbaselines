from collections import OrderedDict
import functools

import torch
from torchvision import datasets, transforms

from olympus.datasets.dataset import AllDataset
from olympus.datasets.transform import minimize


class MNIST(AllDataset):
    """
    Properties
    ----------
    * 60,000 images in the training set
    * 10,000 images in the test set

    Reference
    ---------
    * `Specification <http://yann.lecun.com/exdb/mnist/>`_
    * `wikipedia <https://en.wikipedia.org/wiki/MNIST_database>`_

    Related
    -------
    * Extended MNIST: EMNIST (240 000 + 40 000 images)
    """
    def __init__(self, data_path, mini=False):
        transformations = [
            transforms.Normalize((0.1307,), (0.3081,))
        ]

        if mini:
            transformations.insert(0, minimize(7))

        transform = transforms.Compose(transformations)

        train_dataset = datasets.MNIST(
            data_path, train=True, download=True,
            transform=transforms.ToTensor()
        )
        test_dataset = datasets.MNIST(
            data_path, train=False, download=True,
            transform=transforms.ToTensor()
        )

        super(MNIST, self).__init__(
            torch.utils.data.ConcatDataset([train_dataset, test_dataset]),
            test_size=len(test_dataset),
            transforms=transform
        )


builders = {
    'mnist': MNIST,
    'mini-mnist': functools.partial(MNIST, mini=True)
}
