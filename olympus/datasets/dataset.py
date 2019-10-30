from collections import defaultdict

from torch.utils.data.dataset import Dataset as TorchDataset
from typing import Callable, Tuple


class AllDataset(TorchDataset):
    """Olympus data sets are concatenated data sets that includes train, validation and test sets
    This allow us to change how each sets are splits and give us greater power to design performance
    tests.

    Read more on how Olympus uses custom splits to evaluate model performance at :ref XYZ
    """
    # Underlying Pytorch dataset
    dataset: TorchDataset = None

    # Callable object that apply a transformation on each sample of the data set
    # if you are looking to add data augmentation step you should be looking at
    # ref...
    transforms: Callable = lambda sample: sample

    def __init__(self, dataset, input_shape=None, output_shape=None,
                 train_size=None, valid_size=None, test_size=None, transforms=None):
        self.dataset = dataset
        self._input_shape = input_shape
        self._train_size = train_size
        self._valid_size = valid_size
        self._test_size = test_size
        self._input_shape = input_shape
        self._output_shape = output_shape

        if not isinstance(transforms, dict):
            transforms = dict(train=transforms, valid=transforms, test=transforms)

        if 'valid' not in transforms:
            transforms['valid'] = transforms['test']

        self.transforms = transforms

    @property
    def train_size(self):
        if self._train_size is None:
            return len(self) - self.test_size - self.valid_size

    @property
    def valid_size(self):
        if self._valid_size is None:
            return self.test_size

    @property
    def test_size(self):
        return self._test_size

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
    
    @property
    def input_shape(self):
        if self._input_shape is None:
            return tuple(self.transforms['train'](self.dataset[0][0]).shape)

        return self._input_shape

    @property
    def output_shape(self):
        if self._output_shape is None:
            if isinstance(self.dataset[0][1], int):
                self._output_shape = (len(self.classes), )
            else:
                self._output_shape = self.dataset[0][1].shape

        return self._output_shape

    @property
    def classes(self):
        classes = defaultdict(list)

        for index, [_, y] in enumerate(self.dataset):
            classes[y].append(index)

        return [classes[i] for i in sorted(classes.keys())]
