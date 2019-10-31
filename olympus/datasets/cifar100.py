import torch
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image

from olympus.datasets.dataset import AllDataset


class CIFAR100(AllDataset):
    """
    Properties
    ----------
    * 40,000 images in the training set
    * 10,000 images in the validation set
    * 10,000 images in the test set
    * 100 classes

    Reference
    ---------
    * `Specification <https://www.cs.toronto.edu/~kriz/cifar.html>`_

    """
    def __init__(self, data_path):
        transformations = [
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

        train_transform = [
            to_pil_image,
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()] + transformations

        transformations = dict(
            train=transforms.Compose(train_transform),
            valid=transforms.Compose(transformations),
            test=transforms.Compose(transformations))

        train_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transforms.ToTensor())

        super(CIFAR100, self).__init__(
            torch.utils.data.ConcatDataset([train_dataset, test_dataset]),
            test_size=len(test_dataset),
            transforms=transformations
        )


builders = {
    'cifar100': CIFAR100}
