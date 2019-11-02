import h5py

import numpy

from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    """Dataset wrapping HDF5 tensors."""

    def __init__(self, file_name, transform=None, target_transform=None):
        self.file_name = file_name

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
        return len(h5py.File(self.file_name, 'r', libver='latest', swmr=True)['labels'])
