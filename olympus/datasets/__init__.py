import copy
import os
import torch

from torch.utils.data import DataLoader as TorchDataLoader
from olympus.utils import warning
from olympus.utils.factory import fetch_factories
from olympus.datasets.transform import TransformedSubset
from olympus.datasets.sampling import generate_indices
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


def set_data_path(config):
    if "OLYMPUS_DATA_PATH" not in os.environ:
        print('WARNING: Environment variable OLYMPUS_DATA_PATH is not set. '
              'Data will be downloaded in {}'.format(os.getcwd()))

    config['data_path'] = os.environ.get('OLYMPUS_DATA_PATH', os.getcwd())


# TODO refactor this into something readable
def split_data(datasets, seed, batch_size, sampling_method, num_workers=0):

    sampling_method = copy.deepcopy(sampling_method)
    indices = generate_indices(datasets, sampling_method.pop('name'), **sampling_method)

    data_loaders = dict()

    for split_name, split_indices in indices.items():
        dataset = TransformedSubset(datasets, split_indices, datasets.transforms[split_name])
        sampler = RandomSampler(dataset, seed)

        data_loaders[split_name] = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=type(datasets).collate_fn)

    return data_loaders


def merge_data_loaders(*data_loaders):
    # data_loaders are torch loaders
    for loader in data_loaders:
        assert isinstance(loader, TorchDataLoader)

    # torch loaders have a dataset of TransformedSubset and a sampler of RandomSampler
    data_source = data_loaders[0].dataset.dataset

    # RandomSampler does not have the need the sampler backend have
    seed = data_loaders[0].sampler.sampler.seed

    indices = sum((list(data_loader.dataset.indices) for data_loader in data_loaders), [])

    transform = data_loaders[0].dataset.transform
    batch_size = data_loaders[0].batch_size
    num_workers = data_loaders[0].num_workers

    dataset = TransformedSubset(data_source, indices, transform)
    sampler = RandomSampler(dataset, seed)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return data_loader


def build_loaders(name, sampling_method, seed=1, batch_size=128, num_workers=0, **kwargs):
    set_data_path(kwargs)
    datasets = registered_datasets[name](**kwargs)

    loaders = split_data(
        datasets, seed=seed, batch_size=batch_size, sampling_method=sampling_method,
        num_workers=num_workers)

    return datasets, loaders


class DataLoader:
    def __init__(self, name, sampling_method, seed=1, batch_size=128, num_workers=0, **kwargs):
        self.datasets, self.loaders = build_loaders(
            name,
            sampling_method,
            seed=seed,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )

    def train(self):
        return self.loaders.get('train')

    def valid(self):
        return self.loaders.get('valid')

    def test(self):
        return self.loaders.get('test')

    def get_shapes(self):
        ishape = self.datasets.input_shape
        oshape = self.datasets.target_shape
        return ishape, oshape

    def get_train_valid_loaders(self, hpo_done=False):
        train_loader = self.train()
        valid_loader = self.valid()

        if hpo_done:
            train_loader = merge_data_loaders(train_loader, valid_loader)
            valid_loader = self.test()

        return train_loader, valid_loader
