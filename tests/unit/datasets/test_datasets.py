import pytest

from olympus.datasets import Dataset, SplitDataset, DataLoader
from olympus.datasets.gaussian import Gaussian, Multivariate
from olympus.datasets.tensorhdf5 import HDF5Dataset, generate_hdf5_dataset
from olympus.datasets.imagenet import generate_jpeg_dataset, ImagetNet
from olympus.datasets.tinyimagenet import TinyImageNet
from olympus.datasets.archive import ZipDataset

import os
import shutil

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
    data = Dataset(dataset, path='/tmp/olympus')
    splits = SplitDataset(data, split_method='original')
    loader = DataLoader(
        splits,
        sampler_seed=1,
        batch_size=1
    )

    for i, b in enumerate(loader.train()):
        if i > 10:
            break


def test_gaussian_dataset():
    gaus = Gaussian(size=10, distributions=[
        Multivariate([1, 2], [2, 1]),
        Multivariate([2, 1], [1, 2])])

    print(gaus[0])


def test_hdf5_dataset(file_name='/tmp/hdf5.h5'):
    try:
        total_size = 192
        generate_hdf5_dataset(file_name, (3, 32, 32), num_class=10, samples=total_size)

        hdf5_dataset = HDF5Dataset(file_name)
        assert len(hdf5_dataset) == total_size

        for i in range(total_size):
            img, target = hdf5_dataset[i]
            print(img.shape)

    finally:
        os.remove(file_name)


def test_jpeg_dataset(folder='/tmp/img_folder'):
    try:
        os.makedirs(folder, exist_ok=True)

        generate_jpeg_dataset(f'{folder}/train', (3, 32, 32), num_class=10, samples=128)
        generate_jpeg_dataset(f'{folder}/val', (3, 32, 32), num_class=10, samples=32)
        # generate_jpeg_dataset(f'{folder}/test', (3, 32, 32), num_class=10, samples=32)

        imgnet = ImagetNet(folder)
        assert len(imgnet) == 128 + 32

        for img, target in imgnet:
            pass

    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_tiny_imagenet(folder='/tmp/tiny'):
    try:
        os.makedirs(folder, exist_ok=True)

        generate_hdf5_dataset(
            f'{folder}/tinyimagenet_train.h5', (3, 32, 32), num_class=10, samples=128)
        generate_hdf5_dataset(
            f'{folder}/tinyimagenet_val.h5', (3, 32, 32), num_class=10, samples=32)

        tiny = TinyImageNet(folder)
        for img, target in tiny:
            pass

    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_archive(folder='/tmp/archive'):
    import subprocess

    try:
        os.makedirs(folder, exist_ok=True)

        generate_jpeg_dataset(f'{folder}/train', (3, 32, 32), num_class=10, samples=128)

        subprocess.check_call(f'cd {folder} && zip -r train.zip train', shell=True)
        datasets = ZipDataset(f'{folder}/train.zip')

        for img, target in datasets:
            print(target, end=' ')

    finally:
        shutil.rmtree(folder, ignore_errors=True)
