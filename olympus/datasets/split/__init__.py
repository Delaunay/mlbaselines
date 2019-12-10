from olympus.utils.factory import fetch_factories


sampling_methods = fetch_factories('olympus.datasets.split', __file__, function_name='split')


def generate_splits(datasets, split_method='original', seed=0, ratio=0.1, index=None, data_size=None):
    """Generate splits of a data set using the specified method

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
    """
    if data_size is not None:
        data_size = int(data_size * len(datasets))
        assert data_size <= len(datasets)
    else:
        data_size = len(datasets)

    return sampling_methods[split_method](datasets, data_size, seed, ratio, index)
