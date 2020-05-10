import copy
from collections import defaultdict
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset, Subset, ConcatDataset

from olympus.utils import warning, option, MissingArgument
from olympus.utils.factory import fetch_factories
from olympus.datasets.transformed import TransformedSubset
from olympus.datasets.split import generate_splits
from olympus.datasets.sampling import RandomSampler, SequentialSampler


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


class NotResumable(Exception):
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

    def get_collate_to_dict(self):
        if hasattr(type(self.dataset), 'collate_to_dict'):
            return type(self.dataset).collate_to_dict

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
            *args, **kwargs)

    def load_state_dict(self, states, strict=False):
        # Pytorch creates a sampler even when BatchSampler is given
        sampler = self.loader.sampler
        batch_sampler = self.loader.batch_sampler
        transform = self.loader.dataset.transform

        if transform is not None:
            transform.load_state_dict(states['transform'])

        if sampler is not None and hasattr(sampler, 'load_state_dict'):
            sampler.load_state_dict(states['sampler'])

        elif batch_sampler is not None and hasattr(batch_sampler.sampler, 'load_state_dict'):
            batch_sampler.sampler.load_state_dict(states['sampler'])

        else:
            raise NotResumable('Your sampler is not resumable')

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = dict()

        # Batch sampler is always there
        batch_sampler = self.loader.batch_sampler
        state['sampler'] = batch_sampler.sampler.state_dict()

        transform = self.loader.dataset.transform
        if transform is not None:
            state['transform'] = transform.state_dict()

        return state

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, item):
        if hasattr(self.loader, item):
            return getattr(self.loader, item)

        return super(ResumableDataLoader, self).__getattr__(item)


class DataLoader:
    """Initialize multiple pyTorch DataLoader using a split data set.
    This class holds common arguments use to initialize dataloaders for train, valid and test sets.

    The data loader for a specific set can be retrieved using ``train()``, ``valid()`` and ``test()``
    Additionally, user can override an argument for a specific set.

    Notes
    -----

    Because DataLoader is building new loaders everytime, samplers need to be function calls instead fo actual instance.

    Examples
    --------
    >>> datasets = SplitDataset(
    ...     Dataset('fake_mnist', path='/tmp/mnist'),
    ...     split_method='original')
    >>>
    >>> loader = DataLoader(datasets, sampler_seed=0, batch_size=128)
    >>>
    >>> # Use the constructor default arguments
    >>> train_loader = loader.train()
    >>> print(train_loader.batch_size)
    128
    >>> # Override DataLoader attribute for a specific loader
    >>> test_loader = loader.test(batch_size=256)
    >>> print(test_loader.batch_size)
    256
    >>> # Specify a specific transform per loader
    >>> valid_loader = loader.valid(batch_size=256)
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

    def get_loaders(self, hpo_done=False, transform=None, collate_fn=None, **kwargs):
        """For the final train session we merge train and valid together
        This is an helper function to get the splits needed depending on the context
        """
        if hpo_done:
            train_loader = self.extended_train(transform, collate_fn, **kwargs)
            valid_loader = None
            test_loader = self.test(transform, collate_fn, **kwargs)
        else:
            train_loader = self.train(transform, collate_fn, **kwargs)
            valid_loader = self.valid(transform, collate_fn, **kwargs)
            test_loader = self.test(transform, collate_fn, **kwargs)

        return train_loader, valid_loader, test_loader

    def _get_dataloader(self, subset_name, transform=None, collate_fn=None, **kwargs):
        """Only create them when necessary"""
        if subset_name not in self.loaders:
            self.loaders[subset_name] = self._make_dataloader(
                subset_name, transform=transform, collate_fn=collate_fn, **kwargs)

        return self.loaders[subset_name]

    def _collate_to_dict(self, collate, dataset):
        """Wraps collate function to return a dictionary instead of a tuple
        The mapping is given by the dataset itself through the function ``collate_to_dict``"""
        from torch.utils.data.dataloader import default_collate

        if collate is None:
            collate = default_collate

        if not hasattr(dataset, 'get_collate_to_dict'):
            return collate

        collate_to_dict = dataset.get_collate_to_dict()
        if collate_to_dict is None:
            return collate

        def new_collate(batch):
            result = collate(batch)
            return collate_to_dict(result)

        return new_collate

    def _fetch_collate_function(self, collate_fn, dataset):
        # collate_fn = arguments.get('collate_fn', None)

        # check if the dataset has a special collate function
        if collate_fn is None and hasattr(self.split_dataset, 'get_collate_fn'):
            collate_fn = self.split_dataset.get_collate_fn()

        # wraps our collate function to return dictionaries
        return self._collate_to_dict(collate_fn, dataset)

    def _fetch_sampler(self, sampler, batch_sampler, dataset):
        # Batch sampler holds the sampler & the sample is the one with the random state
        if batch_sampler is not None:
            batch_sampler = batch_sampler(dataset, self.sampler_seed)
            return 'batch_sampler', batch_sampler

        # Use our own sampler if no sampler is set
        if sampler is None:
            sampler = RandomSampler

        return 'sampler', sampler(dataset, self.sampler_seed)

    def _make_dataloader(self, subset_name, **kwargs):
        original_dataset = self.split_dataset.dataset

        arguments = copy.deepcopy(self.default_dataloader_args)
        arguments.update(kwargs)

        if 'batch_sampler' in kwargs and 'batch_sampler' not in self.default_dataloader_args:
            # batch_sampler do not need those args anymore
            arguments.pop('batch_size', None)
            arguments.pop('sampler', None)
            arguments.pop('drop_last', None)
            arguments.pop('shuffle', None)

        transform = arguments.pop('transform', None)
        if transform is None:
            transform = self.split_dataset.transforms.get(subset_name, None)

        arguments['collate_fn'] = self._fetch_collate_function(
            arguments.get('collate_fn', None),
            original_dataset)

        # Make a transformed split for the given subset
        dataset_subset = TransformedSubset(
            # Use the original dataset which has a compliant Dataset interface
            original_dataset,
            getattr(self.split_dataset, subset_name).indices,
            transform)

        valid_batch_size = arguments.pop('valid_batch_size', None)
        if subset_name in ['valid', 'test'] and valid_batch_size:
            arguments['batch_size'] = valid_batch_size

        name, sampler = self._fetch_sampler(
            arguments.get('sampler'),
            arguments.get('batch_sampler'),
            dataset_subset)

        arguments[name] = sampler

        return ResumableDataLoader(
            dataset=dataset_subset,
            **arguments)
