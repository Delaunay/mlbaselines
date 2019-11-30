import pytest

from olympus.datasets import DataLoader

datasets = [
    # 'svhn',
    # 'cifar100',
    # 'pennfudan',
    # 'fashion_mnist',
    # 'cifar10',
    'fake_imagenet',
    'fake_mnist',
    'fake_cifar',
    # 'balanced_emnist',
    # 'mnist',
    # 'mini-mnist',
    # 'test-mnist'
]


@pytest.mark.parametrize('dataset', datasets)
def test_build_dataset(dataset):
    loader = DataLoader(
        dataset,
        seed=0,
        sampling_method={'name': 'original'},
        batch_size=1)

    for i, b in enumerate(loader.train()):
        if i > 10:
            break
