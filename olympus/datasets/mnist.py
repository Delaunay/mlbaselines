import functools

from filelock import FileLock
import torch
from torchvision import datasets, transforms

from olympus.datasets.dataset import AllDataset
from olympus.transforms import minimize
from olympus.utils import option


class MNIST(AllDataset):
    """The MNIST database (Modified National Institute of Standards and Technology database)
    is a large database of handwritten digits that is commonly used for training various image processing systems.
    The database is also widely used for training and testing in the field of machine learning.
    More on `wikipedia <https://en.wikipedia.org/wiki/MNIST_database>`_.

    The full specification can be found at `here <http://yann.lecun.com/exdb/mnist/>`_.
    See also :class:`.BalancedEMNIST` and :class:`.FashionMNIST`

    Attributes
    ----------
    classes: List[int]
        Return the mapping between samples index and their class

    input_shape: (28, 28)
        Size of a sample returned after transformation

    target_shape: (10,)
        The classes are numbers from 0 to 9

    train_size: 50000
        Size of the train dataset

    valid_size: 10000
        Size of the validation dataset

    test_size: 10000
        Size of the test dataset

    References
    ----------
    .. [1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition."
            Proceedings of the IEEE, 86(11):2278-2324, November 1998.

    """
    def __init__(self, data_path, mini=False, train_size=None, valid_size=None, test_size=None,
                 input_shape=None, target_shape=None, **kwargs):
        transformations = [
            transforms.Normalize((0.1307,), (0.3081,))
        ]

        if mini:
            transformations.insert(0, minimize(7))

        transform = transforms.Compose(transformations)

        with FileLock('mnist.lock', timeout=option('download.lock.timeout', 4 * 60, type=int)):
            train_dataset = datasets.MNIST(
                data_path, train=True, download=True,
                transform=transforms.ToTensor()
            )

        with FileLock('mnist.lock', timeout=option('download.lock.timeout', 4 * 60, type=int)):
            test_dataset = datasets.MNIST(
                data_path, train=False, download=True,
                transform=transforms.ToTensor()
            )

        if test_size is None:
            test_size = len(test_dataset)

        super(MNIST, self).__init__(
            torch.utils.data.ConcatDataset([train_dataset, test_dataset]),
            test_size=test_size,
            train_size=train_size,
            valid_size=valid_size,
            transforms=transform,
            input_shape=input_shape,
            target_shape=target_shape
        )

    @staticmethod
    def categories():
        return set(['classification'])


builders = {
    'mnist': MNIST,
    'mini-mnist': functools.partial(MNIST, mini=True),
    # Technically this should be in the sampler logic but we want to test the usual split method
    'test-mnist': functools.partial(MNIST,
                                    train_size=128, valid_size=64, test_size=64,
                                    input_shape=(28, 28), target_shape=(10,)),
}
