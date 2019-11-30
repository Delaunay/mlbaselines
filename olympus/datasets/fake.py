import functools

import numpy

from torchvision import transforms
from torchvision.datasets.fakedata import FakeData

from olympus.datasets.dataset import AllDataset


def default_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])


class FakeDataset(AllDataset):
    """Generate random tensors as input data"""
    def __init__(self, input_shape, target_shape, train_size=1024, valid_size=128, test_size=128, data_path=None):
        if not isinstance(target_shape, int):
            target_shape = numpy.product(target_shape)

        super(FakeDataset, self).__init__(
            FakeData(
                size=train_size + valid_size + test_size,
                image_size=input_shape,
                num_classes=target_shape,
            ),
            test_size=test_size,
            train_size=train_size,
            valid_size=valid_size,
            transforms=default_transform(),
            input_shape=input_shape,
            target_shape=target_shape
        )


builders = {
    'fake_imagenet': functools.partial(FakeDataset, input_shape=(3, 224, 224), target_shape=1000),
    'fake_mnist': functools.partial(FakeDataset, input_shape=(28, 28), target_shape=10),
    'fake_cifar': functools.partial(FakeDataset, input_shape=(3, 32, 32), target_shape=10),
}