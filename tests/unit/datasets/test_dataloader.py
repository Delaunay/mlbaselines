import pytest

import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import BatchSampler

from olympus.datasets import register_dataset, Dataset, SplitDataset, DataLoader
from olympus.datasets.dataset import AllDataset


SIZE = 128


class BaseDataset(AllDataset):
    def __init__(self, **kwargs):
        super(BaseDataset, self).__init__(self, train_size=SIZE, valid_size=0, test_size=0)
        self.data = [(i, i * i) for i in range(SIZE)]

    def __getitem__(self, item):
        return self.data[item]


class DatasetCollate(AllDataset):
    def __init__(self, **kwargs):
        super(DatasetCollate, self).__init__(self, train_size=SIZE, valid_size=0, test_size=0)
        self.data = [(i, i * i, i * i * i) for i in range(SIZE)]

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def collate_fn(data):
        return default_collate([i[:2] for i in data])


class DatasetCollateToDict(AllDataset):
    def __init__(self, **kwargs):
        super(DatasetCollateToDict, self).__init__(self, train_size=SIZE, valid_size=0, test_size=0)
        self.data = [(i, i * i, i * i * i) for i in range(SIZE)]

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def collate_fn(data):
        return default_collate([i[:2] for i in data])


register_dataset('BaseDataset', BaseDataset)
register_dataset('DatasetCollate', DatasetCollate)
register_dataset('DatasetCollateToDict', DatasetCollateToDict)


def make_loader(dataset):
    data = Dataset(dataset, path='/tmp/olympus')
    splits = SplitDataset(data, split_method='original')
    return DataLoader(
        splits,
        sampler_seed=1,
        batch_size=8).train()


def make_loader_batch_sampler(dataset):
    from olympus.datasets.sampling import RandomSampler

    data = Dataset(dataset, path='/tmp/olympus')
    splits = SplitDataset(data, split_method='original')

    sampler = lambda dataset, seed: BatchSampler(
        sampler=RandomSampler(dataset, seed),
        batch_size=8,
        drop_last=True)

    return DataLoader(
        splits,
        sampler_seed=1,
        batch_sampler=sampler).train()


dataset_overrides = [
    ('BaseDataset', 'batch'),
    ('DatasetCollate', 'batch'),
    ('DatasetCollateToDict', 'x')
]


@pytest.mark.parametrize('dataset,name', dataset_overrides)
def test_dataloader_dataset_overrides(dataset, name):
    for sample in make_loader(dataset):
        assert isinstance(sample[0][0], torch.Tensor)
        assert len(sample) == 2
        assert len(sample[0][0]) == 8


@pytest.mark.parametrize('loader_factory', [make_loader, make_loader_batch_sampler])
def test_dataloader_is_resumable(loader_factory):
    def run_dataloader_nointerruption():
        loader = loader_factory('BaseDataset')

        i, batch = -1, None
        for i, batch in enumerate(loader):
            pass

        for i, batch in enumerate(loader):
            pass

        return batch, loader.state_dict()

    def resume_dataloader(state=None):
        loader = loader_factory('BaseDataset')

        if state is not None:
            loader.load_state_dict(state)

        i, batch = -1, None
        for i, batch in enumerate(loader):
            pass

        return batch, loader.state_dict()

    # Epoch 1 & 2
    final_batch, state = run_dataloader_nointerruption()

    # Epoch 1
    _, state = resume_dataloader()

    # Epoch 2
    resumed_batch, resumed_state = resume_dataloader(state)

    assert (resumed_state['sampler']['rng_state'][1] == state['sampler']['rng_state'][1]).all()
    assert (resumed_batch[0][0] - final_batch[0][0]).abs().sum().sum() < 1e-4
