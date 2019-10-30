import torch
from torchvision import datasets, transforms
from olympus.datasets.dataset import AllDataset


class BalancedEMNIST(AllDataset):
    """
    Properties
    ----------
    * 94,000 images in the training set
    * 18,800 images in the validation set
    * 18,800 images in the test set
    * 47 classes

    Reference
    ---------
    * `Specification <https://arxiv.org/abs/1702.05373>`_

    """
    def __init__(self, data_path):
        train_dataset = datasets.EMNIST(
            data_path, train=True, download=True, split='balanced',
            transform=transforms.ToTensor())

        test_dataset = datasets.EMNIST(
            data_path, train=False, download=True, split='balanced',
            transform=transforms.ToTensor())

        super(BalancedEMNIST, self).__init__(
            torch.utils.data.ConcatDataset([train_dataset, test_dataset]),
            test_size=len(test_dataset))


builders = {
    'balanced_emnist': BalancedEMNIST}
