import torch
from torchvision import datasets, transforms

from olympus.datasets.dataset import AllDataset


class FashionMNIST(AllDataset):
    """
    Properties
    ----------
    * 50,000 images in the training set
    * 10,000 images in the validation set
    * 10,000 images in the test set

    Reference
    ---------
    * `Specification <https://github.com/zalandoresearch/fashion-mnist>`_

    """
    def __init__(self, data_path):
        train_dataset = datasets.MNIST(
            data_path, train=True, download=True,
            transform=transforms.ToTensor()
        )

        test_dataset = datasets.MNIST(
            data_path, train=False, download=True,
            transform=transforms.ToTensor()
        )

        super(FashionMNIST, self).__init__(
            torch.utils.data.ConcatDataset([train_dataset, test_dataset]),
            test_size=len(test_dataset)
        )


builders = {
    'fashion_mnist': FashionMNIST}
