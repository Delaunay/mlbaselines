import torch
from torchvision import datasets, transforms
from olympus.datasets.dataset import AllDataset


class BalancedEMNIST(AllDataset):
    """The MNIST database was derived from a larger dataset known as the NIST Special Database 19 which contains digits,
    uppercase and lowercase handwritten letters. This paper introduces a variant of the full NIST dataset,
    which we have called Extended MNIST (EMNIST), which follows the same conversion paradigm used to create
    the MNIST dataset. The result is a set of datasets that constitute a more challenging classification
    tasks involving letters and digits.
    More on `arxiv <https://arxiv.org/abs/1702.05373>`_.

    See also :class:`.MNIST` and :class:`.FashionMNIST`

    Attributes
    ----------
    classes: List[int]
        Return the mapping between samples index and their class

    input_shape: (28, 28)
        Size of a sample stored in this dataset

    target_shape: (47,)
        The dataset is composed of 47 classes, 10 digits, 37 letters

    train_size: 94000
        Size of the train dataset

    valid_size: 18800
        Size of the validation dataset

    test_size: 18800
        Size of the test dataset

    References
    ----------
    .. [1] Gregory Cohen, Saeed Afshar, Jonathan Tapson, André van Schaik.
        "EMNIST: an extension of MNIST to handwritten letters", Mar 2017

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

    @staticmethod
    def categories():
        return set(['classification'])


builders = {
    'balanced_emnist': BalancedEMNIST}
