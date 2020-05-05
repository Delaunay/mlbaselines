from filelock import FileLock

import numpy

import torch
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image

from olympus.datasets.transformed import Compose, RandomCrop, RandomHorizontalFlip
from olympus.datasets.dataset import AllDataset
from olympus.utils import option


class CIFAR10(AllDataset):
    """The CIFAR-10 dataset (Canadian Institute For Advanced Research) is a collection of images
    that are commonly used to train machine learning and computer vision algorithms.
    It is one of the most widely used datasets for machine learning research.
    The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes.
    More on `wikipedia <https://en.wikipedia.org/wiki/CIFAR-10>`_.

    The full specification can be found at `here <https://www.cs.toronto.edu/~kriz/cifar.html>`_.
    See also :class:`.CIFAR100`

    Attributes
    ----------
    classes: List[int]
        Return the mapping between samples index and their class

    input_shape: (3, 32, 32)
        Size of a sample stored in this dataset

    target_shape: (10,)
        There are 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

    train_size: 40000
        Size of the train dataset

    valid_size: 10000
        Size of the validation dataset

    test_size: 10000
        Size of the test dataset

    References
    ----------
    .. [1] Alex Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", 2009.

    """
    def __init__(self, data_path, transform=True, transform_seed=0, cache=None):
        transformations = [
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

        rng = numpy.random.RandomState(transform_seed)

        if transform:
            train_transform = [
                to_pil_image,
                RandomCrop(32, padding=4, seed=rng.randint(2**30)),
                RandomHorizontalFlip(seed=rng.randint(2**30)),
                transforms.ToTensor()] + transformations

        else:
            train_transform = transformations

        transformations = dict(
            train=Compose(train_transform),
            valid=Compose(transformations),
            test=Compose(transformations))

        with FileLock('cifar10.lock', timeout=option('download.lock.timeout', 4 * 60, type=int)):
            train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())

        with FileLock('cifar10.lock', timeout=option('download.lock.timeout', 4 * 60, type=int)):
            test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transforms.ToTensor())

        super(CIFAR10, self).__init__(
            torch.utils.data.ConcatDataset([train_dataset, test_dataset]),
            test_size=len(test_dataset),
            transforms=transformations
        )

    @staticmethod
    def categories():
        return set(['classification'])


builders = {
    'cifar10': CIFAR10}
