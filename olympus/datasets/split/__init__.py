from torch.utils.data import Dataset as TorchDataset, Subset

from olympus.utils import new_seed
from olympus.utils.factory import fetch_factories


sampling_methods = fetch_factories('olympus.datasets.split', __file__, function_name='split')


def known_split_methods():
    return list(sampling_methods.keys())


class RegisteredSplitMethodNotFound(Exception):
    pass


def _generate_splits(datasets, split_method, seed, ratio, index, data_size, balanced):
    if data_size is not None:
        data_size = int(data_size * len(datasets))
        assert data_size <= len(datasets)
    else:
        data_size = len(datasets)

    split = sampling_methods.get(split_method)

    if split is None:
        raise RegisteredSplitMethodNotFound(
            f'Split method `{split_method}` was not found use {known_split_methods()}')

    return split(datasets, data_size, seed, ratio, index, balanced)


class SplitDataset(TorchDataset):
    """Split the main dataset into 3 subsets using the split_method
    Generate splits of a data set using the specified method

    Attributes
    ----------
    seed: int
        Seed of the PRNG, when the split use split to generate the splits

    ratio: float
        Split Ratio for the test and validation test (default: 0.1 i.e 10%)

    data_size: int
        Specify the number of points. It defaults to the full size of the data set.
        if it is specified then data_size

    index: int
        If data size is small enough, multiple splits of the same data set can be extracted.
        index specifies which of those splits is fetched

    balanced: bool
        If true, the splits will keep the classes balanced.
    """
    def __init__(self, dataset, split_method='original', seed=new_seed(split=0), ratio=0.1, index=None,
                 data_size=None, balanced=False):
        self.dataset = dataset

        if isinstance(split_method, dict):
            kwargs = split_method
        else:
            kwargs = dict(
                split_method=split_method,
                seed=seed,
                ratio=ratio,
                index=index,
                data_size=data_size,
                balanced=balanced)

        self.splits = _generate_splits(dataset, **kwargs)

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
