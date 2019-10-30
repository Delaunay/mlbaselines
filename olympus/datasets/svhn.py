import torch
from torchvision import datasets, transforms

from olympus.datasets.dataset import AllDataset


class SVHN(AllDataset):
    """
    Properties
    ----------
    * 47225 images in the training set
    * 26032 images in the validation set
    * 26032 images in the test set

    Reference
    ---------
    * `Specification <http://ufldl.stanford.edu/housenumbers/>`_

    """
    def __init__(self, data_path):
        train_dataset = datasets.SVHN(
            data_path, split='train', download=True,
            transform=transforms.ToTensor())

        test_dataset = datasets.SVHN(
            data_path, split='test', download=True,
            transform=transforms.ToTensor())

        super(SVHN, self).__init__(
            torch.utils.data.ConcatDataset([train_dataset, test_dataset]),
            test_size=len(test_dataset)
        )


builders = {
    'svhn': SVHN}
