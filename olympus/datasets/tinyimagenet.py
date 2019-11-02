from collections import OrderedDict
import csv
import functools
import os
import urllib
import zipfile
import shutil
import time

from filelock import FileLock, Timeout

from PIL import Image

import h5py

import numpy

from tqdm import tqdm

import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as F

from olympus.datasets.dataset import AllDataset
from olympus.datasets.tensorhdf5 import HDF5Dataset
from olympus.datasets.transform import to_pil_image

# download-url: http://cs231n.stanford.edu/tiny-imagenet-200.zip

# Train: 100000
# Val:    10000
# Train:  10000

DIRNAME = 'tiny-imagenet-200'
ZIP_FILENAME = 'tiny-imagenet-200.zip'
TRAIN_FILENAME = 'tinyimagenet_train.h5'
VAL_FILENAME = 'tinyimagenet_val.h5'
# TEST_FILENAME = 'tinyimagenet_test.h5'


def get_zipfile_path(data_path):
    return os.path.join(data_path, ZIP_FILENAME)


def get_dirpath(data_path):
    return os.path.join(data_path, DIRNAME)


def all_hdf5_exists(data_path):
    return all(os.path.exists(os.path.join(data_path, filename))
               for filename in [TRAIN_FILENAME, VAL_FILENAME])


def build_dataset(data_path, timeout=10 * 60):
    if all_hdf5_exists(data_path):
        return

    try:
        with FileLock(os.path.join(data_path, DIRNAME + ".lock"), timeout=timeout):
            download(data_path)
            unzip(data_path)
            create_hdf5(data_path)
    except Timeout:
        print("Another process holds the lock since more than {} seconds. "
              "Will try to load the dataset.").format(timeout)
    finally:
        clean(data_path)


def download(data_path):
    if os.path.exists(get_zipfile_path(data_path)):
        print("Zip file already downloaded")
        return

    # download
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    u = urllib.request.urlopen(url)
    with open(get_zipfile_path(data_path), 'wb') as f:
        file_size = int(dict(u.getheaders())['Content-Length']) / (10.0**6)
        print("Downloading: {} ({}MB)".format(get_zipfile_path(data_path), file_size))

        file_size_dl = 0
        block_sz = 8192
        pbar = tqdm(total=file_size, desc='TinyImageNet')
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            f.write(buffer)
            pbar.update(len(buffer) / (10.0 ** 6))

        pbar.close()


def unzip(data_path):
    print("Unzipping files...")
    with zipfile.ZipFile(get_zipfile_path(data_path), 'r') as zip_ref:
        zip_ref.extractall(data_path)
    print("Done")


def clean(data_path):
    print("Deleting unzipped files...")
    shutil.rmtree(get_dirpath(data_path))


def create_hdf5(data_path):
    create_hdf5_train(
        get_dirpath(data_path), os.path.join(data_path, 'tinyimagenet_train.h5'))

    create_hdf5_val(
        get_dirpath(data_path), os.path.join(data_path, 'tinyimagenet_val.h5'))


def create_train_loader(dirpath):
    dataset = datasets.ImageFolder(
        os.path.join(dirpath, 'train'),
        transforms.Compose([transforms.ToTensor()]))

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1, num_workers=1)

    for batch in dataloader:
        yield batch


def create_hdf5_file(dirpath, file_path, n, dataloader):

    f = h5py.File(file_path, 'w', libver='latest')

    data = f.create_dataset(
        "data", (n, 64, 64, 3),
        chunks=(1, 64, 64, 3),
        dtype=numpy.uint8)
        # compression='lzf')
    labels = f.create_dataset("labels", (n, ), dtype=numpy.uint8)

    f.swmr_mode = True

    for index, (x, y) in enumerate(tqdm(dataloader, total=n, desc='HDF5')):
        x = numpy.array(x * 255, dtype=numpy.uint8)
        data[index] = numpy.moveaxis(x, 1, -1)
        labels[index] = y

    f.close()


def create_hdf5_train(dirpath, file_path):
    return create_hdf5_file(dirpath, file_path, 100000, create_train_loader(dirpath))


def create_hdf5_val(dirpath, file_path):
    return create_hdf5_file(dirpath, file_path, 10000, create_val_loader(dirpath))


def create_val_loader(dirpath):

    train_dataset = datasets.ImageFolder(
        os.path.join(dirpath, 'train'),
        transforms.Compose([transforms.ToTensor()]))

    with open(os.path.join(dirpath, 'val', 'val_annotations.txt'), 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t')

        for index, row in enumerate(csv_reader):
            filename = row[0]
            class_id = row[1]

            image_path = os.path.join(dirpath, 'val', 'images', filename)
            with open(image_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                x = F.to_tensor(img).unsqueeze(0)

            yield x, train_dataset.class_to_idx[class_id]


class TinyImageNet(AllDataset):
    """Tiny Imagenet has 200 classes. Each class has 500 training images, 50 validation images, and 50 test images.
    We have released the training and validation sets with images and annotations.
    We provide both class labels and bounding boxes as annotations;
    however, you are asked only to predict the class label of each image without localizing the objects.
    The test set is released without labels. More at `tiny-imagenet <https://tiny-imagenet.herokuapp.com/>`_.

    Attributes
    ----------
    classes: List[int]
        Return the mapping between samples index and their class

    input_shape: (3, 64, 64)
        Size of a sample stored in this dataset

    target_shape: (200,)
        The dataset is composed of 200 classes

    train_size: 90000
        Size of the train dataset

    valid_size: 10000
        Size of the validation dataset

    test_size: 10000
        Size of the test dataset

    References
    ----------
    .. [1] Jiayu Wu, Qixiang Zhang, Guoxi Xu. "Tiny ImageNet Challenge", 2017

    """
    def __init__(self, data_path):
        build_dataset(data_path)

        base_transformations = transforms.Compose([
            # data is stored as uint8
            to_pil_image,
            transforms.CenterCrop(64),
            transforms.ToTensor()])

        transformations = [
            transforms.Normalize(mean=[0.4194, 0.3898, 0.3454],
                                 std=[0.303, 0.291, 0.293])]

        train_transform = [
            to_pil_image,
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ] + transformations

        transformations = dict(
            train = transforms.Compose(train_transform),
            valid = transforms.Compose(transformations),
            test = transforms.Compose(transformations))

        train_dataset = HDF5Dataset(
            os.path.join(data_path, TRAIN_FILENAME),
            base_transformations,
            transforms.Lambda(lambda x: int(x)))

        test_dataset = HDF5Dataset(
            os.path.join(data_path, VAL_FILENAME),
            base_transformations,
            transforms.Lambda(lambda x: int(x)))

        super(TinyImageNet, self).__init__(
            torch.utils.data.ConcatDataset([train_dataset, test_dataset]),
            test_size=len(test_dataset),
            transforms=transformations,
            output_shape=(200, ),
        )


builders = {
    'tinyimagenet': TinyImageNet}


if __name__ == "__main__":
    from bvdl.utils.cov import ExpectationMeter, CovarianceMeter
    for num_workers in range(1, 9): # range(5, 6):  # 1, 9):
        print("\n-*- {} -*-\n".format(num_workers))
        datasets = build(128, "/Tmp/data", num_workers)
        std = CovarianceMeter()
        topmax = 0
        for x, y in tqdm(datasets['train'], desc='train'):
            flattened = x.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            std.add(flattened, n=flattened.size(0))
            topmax = max(topmax, y.max())
        print(topmax)
        print(std.expectation_meter.value())
        print(std.value())
