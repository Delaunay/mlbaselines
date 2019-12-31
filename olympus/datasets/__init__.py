import copy
from collections import defaultdict
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset, Subset, ConcatDataset

from olympus.utils import warning, option, MissingArgument
from olympus.utils.factory import fetch_factories
from olympus.datasets.transformed import TransformedSubset
from olympus.datasets.split import generate_splits
from olympus.datasets.sampling import RandomSampler


registered_datasets = fetch_factories('olympus.datasets', __file__)


def known_datasets(*category_filters, include_unknown=False):
    if not category_filters:
        return registered_datasets.keys()

    matching = []
    for filter in category_filters:
        for name, factory in registered_datasets.items():

            if hasattr(factory, 'categories'):
                if filter in factory.categories():
                    matching.append(name)

            # we don't know if it matches because it does not have the categories method
            elif include_unknown:
                matching.append(name)

    return matching


def register_dataset(name, factory, override=False):
    global registered_datasets

    if name in registered_datasets:
        warning(f'{name} was already registered, use override=True to ignore')

        if not override:
            return

    registered_datasets[name] = factory


class RegisteredDatasetNotFound(Exception):
    pass


class Dataset(TorchDataset):
    """Public Interface of the Dataset"""
    def __init__(self, name=None, dataset=None, path=option('data.path', default='/tmp/olympus/data'), **kwargs):
        if dataset is not None:
            self.dataset = dataset

        elif name is not None:
            dataset_ctor = registered_datasets.get(name)

            if dataset_ctor is None:
                raise RegisteredDatasetNotFound(name)

            self.dataset = dataset_ctor(data_path=path, **kwargs)

        else:
            raise MissingArgument('Dataset or Name need to be set')

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    @property
    def transforms(self):
        if hasattr(self.dataset, 'transforms'):
            return self.dataset.transforms
        return {}

    @property
    def input_shape(self):
        if hasattr(self.dataset, 'input_shape'):
            return self.dataset.input_shape

    @property
    def target_shape(self):
        if hasattr(self.dataset, 'target_shape'):
            return self.dataset.target_shape

    def get_collate_fn(self):
        if hasattr(type(self.dataset), 'collate_fn'):
            return type(self.dataset).collate_fn

        return None

    @property
    def train_size(self):
        """Size of the training set"""
        if hasattr(self.dataset, 'train_size'):
            return self.dataset.train_size

    @property
    def valid_size(self):
        """Size of the validation set"""
        if hasattr(self.dataset, 'valid_size'):
            return self.dataset.valid_size

    @property
    def test_size(self):
        """Size of the test set"""
        if hasattr(self.dataset, 'test_size'):
            return self.dataset.test_size

    @property
    def classes(self):
        """Return the mapping between samples index and their class"""
        classes = defaultdict(list)

        for index, [_, y] in enumerate(self.dataset):
            classes[y].append(index)

        return [classes[i] for i in sorted(classes.keys())]

    def categories(self):
        """Dataset tags so we can filter what we want depending on the task"""
        if hasattr(self.dataset, 'categories'):
            return self.dataset.categories

        return set()


class SplitDataset(TorchDataset):
    """Split the main dataset into 3 subsets using the split_method"""
    def __init__(self, dataset, split_method, data_size=None, seed=1, ratio=0.1, index=0):
        self.dataset = dataset
        self.splits = generate_splits(dataset, split_method, seed, ratio, index, data_size)

    # This function is not compliant with the Dataset Interface on purpose
    # we do not want people to use this class as a normal dataset because it is not
    # def __getitem__(self, subset_name, index):
    #   raise getattr(self, subset_name)[index]

    @property
    def train(self) -> Subset:
        return Subset(self.dataset, self.splits.train)

    @property
    def valid(self) -> Subset:
        return Subset(self.dataset, self.splits.valid)

    @property
    def test(self) -> Subset:
        return Subset(self.dataset, self.splits.test)

    @property
    def extended_train(self):
        import numpy as np
        merged_indices = np.concatenate([self.splits.train, self.splits.valid])
        return Subset(self.dataset, merged_indices)

    @property
    def train_indices(self):
        return self.splits.train

    @property
    def test_indices(self):
        return self.splits.test

    @property
    def valid_indices(self):
        return self.splits.valid

    def get_collate_fn(self):
        return self.dataset.get_collate_fn()

    def __getattr__(self, item):
        if hasattr(self.dataset, item):
            return getattr(self.dataset, item)

        raise AttributeError(f'Attribute {item} was not found in SplitDataset')


class ResumableDataLoader:
    def __init__(self, *args, **kwargs):
        self.loader = torch.utils.data.DataLoader(
            *args, **kwargs
        )

    def load_state_dict(self, states, strict=False):
        self.loader.sampler.load_state_dict(states['sampler'])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {
            'sampler': self.loader.sampler.state_dict()
        }

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


class DataLoader:
    """Initialize multiple pyTorch DataLoader using a split data set

    Notes
    -----
    While Olympus could make this DataLoader fall back to the expected pytorch DataLoader behaviour.
    It might be risky to do so as ``train``, ``valid`` and ``test`` would return the same set every time.

    Examples
    --------
    >>> datasets = SplitDataset(
    >>>     Dataset('mnist', path='/tmp/mnist'),
    >>>     split_method='original')
    >>>
    >>> loader = DataLoader(datasets, sampler_seed=0, batch_size=128)
    >>>
    >>> # Use the constructor default arguments
    >>> train_loader = loader.train()
    >>> print(train_loader.batch_size)  # 128
    >>>
    >>> # Override DataLoader attribute for a specific loader
    >>> test_loader = loader.test(batch_size=256)
    >>> print(test_loader.batch_size)   #  256
    >>>
    >>> # Specify a specific transform per loader
    >>> valid_loader = loader.valid(batch_size=256, transforms=...)
    """
    def __init__(self, split_dataset: SplitDataset, sampler_seed, **kwargs):
        self.split_dataset = split_dataset
        self.sampler_seed = sampler_seed
        self.default_dataloader_args = kwargs
        self.loaders = {}

    def train(self, *args, **kwargs):
        """The arguments provided to this function will override the arguments provided in the constructor"""
        return self._get_dataloader('train', *args, **kwargs)

    def valid(self, *args, **kwargs):
        """The arguments provided to this function will override the arguments provided in the constructor"""
        return self._get_dataloader('valid', *args, **kwargs)

    def test(self, *args, **kwargs):
        """The arguments provided to this function will override the arguments provided in the constructor"""
        return self._get_dataloader('test', *args, **kwargs)

    def extended_train(self, *args, **kwargs):
        """The arguments provided to this function will override the arguments provided in the constructor"""
        return self._get_dataloader('extended_train', *args, **kwargs)

    def get_shapes(self):
        ishape = self.split_dataset.input_shape
        oshape = self.split_dataset.target_shape
        return ishape, oshape

    def get_train_valid_loaders(self, hpo_done=False, transform=None, collate_fn=None, **kwargs):
        """For the final train session we merge train and valid together
        This is an helper function to get the splits needed depending on the context
        """
        if hpo_done:
            train_loader = self.extended_train(transform, collate_fn, **kwargs)
            valid_loader = self.test(transform, collate_fn, **kwargs)
        else:
            train_loader = self.train(transform, collate_fn, **kwargs)
            valid_loader = self.valid(transform, collate_fn, **kwargs)

        return train_loader, valid_loader

    def _get_dataloader(self, subset_name, transform=None, collate_fn=None, **kwargs):
        """Only create them when necessary"""
        if subset_name not in self.loaders:
            self.loaders[subset_name] = self._make_dataloader(
                subset_name, transform, collate_fn, **kwargs)

        return self.loaders[subset_name]

    def _make_dataloader(self, subset_name, transform, collate_fn, **kwargs):
        arguments = copy.deepcopy(self.default_dataloader_args)
        arguments.update(kwargs)

        if transform is None:
            transform = self.split_dataset.transforms.get(subset_name, None)

        if transform is None:
            transform = lambda x: x

        if collate_fn is None and hasattr(self.split_dataset, 'get_collate_fn'):
            collate_fn = self.split_dataset.get_collate_fn()

        dataset_subset = TransformedSubset(
            # Use the original dataset which has a compliant Dataset interface
            self.split_dataset.dataset,
            getattr(self.split_dataset, subset_name).indices,
            transform)

        sampler = RandomSampler(dataset_subset, self.sampler_seed)

        return ResumableDataLoader(
            dataset=dataset_subset,
            sampler=sampler,
            collate_fn=collate_fn,
            **arguments
        )
