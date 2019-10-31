import os
import torch

from olympus.utils.factory import fetch_factories
from olympus.datasets.transform import TransformedSubset
from olympus.datasets.sampling import generate_indices


factories = fetch_factories('olympus.datasets', __file__)


def set_data_path(config):
    if "OLYMPUS_DATA_PATH" not in os.environ:
        print('WARNING: Environment variable OLYMPUS_DATA_PATH is not set. '
              'Data will be downloaded in {}'.format(os.getcwd()))

    config['data_path'] = os.environ.get('OLYMPUS_DATA_PATH', os.getcwd())


def split_data(datasets, batch_size, sampling_method, num_workers=0):

    indices = generate_indices(datasets, sampling_method.pop('name'), **sampling_method)

    data_loaders = dict()
    for split_name, split_indices in indices.items():
        dataset = TransformedSubset(datasets, split_indices, datasets.transforms[split_name])
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        data_loaders[split_name] = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return data_loaders


def merge_data_loaders(*data_loaders):
    data_source = data_loaders[0].datasets.datasets
    indices = sum((list(data_loader.datasets.indices) for data_loader in data_loaders), [])
    transform = data_loaders[0].datasets.transform
    batch_size = data_loaders[0].batch_size
    num_workers = data_loaders[0].num_workers

    dataset = TransformedSubset(data_source, indices, transform)
    sampler = torch.utils.data.sampler.RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return data_loader


def build_loaders(name, sampling_method, batch_size=128, num_workers=0, **kwargs):
    set_data_path(kwargs)
    datasets = factories[name](**kwargs)

    loaders = split_data(datasets, batch_size=batch_size, sampling_method=sampling_method,
                         num_workers=num_workers)

    return datasets, loaders
