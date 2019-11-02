import torch
from torchvision import datasets, transforms

from olympus.datasets.dataset import AllDataset


class FashionMNIST(AllDataset):
    """Fashion-MNIST, is a dataset comprising of 28x28 grayscale images of 70,000 fashion products from 10 categories,
    with 7,000 images per category. The training set has 60,000 images and the test set has 10,000 images.
    Fashion-MNIST is intended to serve as a direct drop-in replacement for the original MNIST dataset
    for benchmarking machine learning algorithms, as it shares the same image size,
    data format and the structure of training and testing splits.
    More on `arxiv <https://arxiv.org/abs/1708.07747>`_.

    The full specification can be found at `here <https://github.com/zalandoresearch/fashion-mnist>`_.
    See also :class:`.BalancedEMNIST` and :class:`.MNIST`

    Attributes
    ----------
    classes: List[int]
        Return the mapping between samples index and their class

    dataset: TorchDataset
        Underlying dataset

    input_shape: (28, 28)
        Size of a sample stored in this dataset

    output_shape: (10,)
        The classes are (T-shirt, Trouser, Pullover, Dress, Coat, Sandals, Shirt, Sneaker, Bag, Ankle Boot)

    train_size: 50000
        Size of the train dataset

    valid_size: 10000
        Size of the validation dataset

    test_size: 10000
        Size of the test dataset

    References
    ----------
    .. [1] Han Xiao, Kashif Rasul, Roland Vollgraf.
           "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms"  Aug 2017

    """
    def __init__(self, data_path):
        train_dataset = datasets.FashionMNIST(
            data_path, train=True, download=True,
            transform=transforms.ToTensor()
        )

        test_dataset = datasets.FashionMNIST(
            data_path, train=False, download=True,
            transform=transforms.ToTensor()
        )

        super(FashionMNIST, self).__init__(
            torch.utils.data.ConcatDataset([train_dataset, test_dataset]),
            test_size=len(test_dataset)
        )


builders = {
    'fashion_mnist': FashionMNIST}
