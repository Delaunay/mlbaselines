import h5py

import numpy

from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    """Dataset wrapping HDF5 tensors."""

    def __init__(self, data_path, transform=None, target_transform=None):
        self.file_name = data_path

        self.transform = transform
        self.target_transform = target_transform

        self._file = None
        self._labels = None
        self._data = None

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self.file_name, 'r', libver='latest', swmr=True)

        return self._file

    @property
    def labels(self):
        if self._labels is None:
            self._labels = self.file['labels']

        return self._labels

    @property
    def data(self):
        if self._data is None:
            self._data = self.file['data']

        return self._data

    def __getitem__(self, index):
        # This is only necessary if there is a concurrent writer.
        # self.data.id.refresh()
        sample = self.data[index]
        sample = sample.astype(numpy.uint8)

        if self.transform is not None:
            sample = self.transform(sample)

        target = self.labels[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return h5py.File(self.file_name, 'r', libver='latest', swmr=True)['data'].shape[0]


def generate_hdf5_dataset(file_name, shape=(3, 224, 224), num_class=1000, samples=192):
    """Generate a Fake HDF5 Dataset for testing and benchmarking purposes"""
    from olympus.datasets.fake import FakeDataset

    fake = FakeDataset(shape, num_class, samples, 0, 0)
    fake_shape = shape[1:] + (shape[0],)

    with h5py.File(file_name, 'w', libver='latest', swmr=True) as h5file:
        data = h5file.create_dataset("data", (samples,) + fake_shape, dtype='i')
        labels = h5file.create_dataset("labels", (samples,), dtype='i')

        for i, (image, target) in enumerate(fake):
            data[i, :] = image
            labels[i] = target