from collections import defaultdict

from torch.utils.data.dataset import Dataset as TorchDataset
from typing import Callable


class AllDataset(TorchDataset):
    """Olympus data sets are concatenated data sets that includes train, validation and test sets
    This allow us to change how each sets are splits and give us greater power to design performance
    tests.

    Read more on how Olympus uses custom splits to evaluate model performance at :ref XYZ

    Attributes
    ----------
    dataset: TorchDataset
        Underlying dataset (concatenation of original train and test sets)

    collate_fn: Optional[Callable] !! static method !!
        merges a list of samples to form a mini-batch of Tensor(s).  Used when using batched loading from a
        map-style dataset.
    """
    # Underlying Pytorch dataset
    dataset: TorchDataset = None

    # Callable object that apply a transformation on each sample of the data set
    # if you are looking to add data augmentation step you should be looking at
    # ref...
    transforms: Callable = lambda sample: sample
    collate_fn: Callable = None

    def __init__(self, dataset, data_path=None, input_shape=None, target_shape=None,
                 train_size=None, valid_size=None, test_size=None, transforms=None):
        self.dataset = dataset
        self._input_shape = input_shape
        self._train_size = train_size
        self._valid_size = valid_size
        self._test_size = test_size
        self._input_shape = input_shape
        self._target_shape = target_shape

        if transforms is None:
            transforms = lambda data: data

        if not isinstance(transforms, dict):
            transforms = dict(train=transforms, valid=transforms, test=transforms)

        if 'valid' not in transforms:
            transforms['valid'] = transforms['test']

        self.transforms = transforms

    @property
    def train_size(self):
        """Size of the training set"""
        if self._train_size is None:
            return len(self) - self.test_size - self.valid_size
        return self._train_size

    @property
    def valid_size(self):
        """Size of the validation set"""
        if self._valid_size is None:
            return self.test_size
        return self._valid_size

    @property
    def test_size(self):
        """Size of the test set"""
        return self._test_size

    def __getitem__(self, idx):
        """Return a sample from the entire dataset"""
        return self.dataset[idx]

    def __len__(self):
        """Return the number of samples inside the dataset"""
        if self._train_size is None:
            return len(self.dataset)
        return self.valid_size + self.train_size + self.test_size

    @property
    def input_shape(self):
        """Return the size of the samples"""
        if self._input_shape is None:
            return tuple(self.transforms['train'](self.dataset[0][0]).shape)

        return self._input_shape

    @property
    def target_shape(self):
        """Return the size of the target"""
        if self._target_shape is None:
            if isinstance(self.dataset[0][1], int):
                self._target_shape = (len(self.classes), )
            else:
                self._target_shape = self.dataset[0][1].shape

        return self._target_shape

    @property
    def classes(self):
        """Return the mapping between samples index and their class"""
        classes = defaultdict(list)

        for index, [_, y] in enumerate(self.dataset):
            classes[y].append(index)

        return [classes[i] for i in sorted(classes.keys())]

    @staticmethod
    def categories():
        """Dataset tags so we can filter what we want depending on the task"""
        return set()
