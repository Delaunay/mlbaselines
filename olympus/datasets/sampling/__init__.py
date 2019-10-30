from olympus.utils.factory import fetch_factories


sampling_methods = fetch_factories('olympus.datasets.sampling', __file__, function_name='sample')


def generate_indices(datasets, sampling_method='original', data_size=None, **kwargs):
    if data_size is not None:
        data_size = int(data_size * len(datasets))
        assert data_size <= len(datasets)
    else:
        data_size = len(datasets)

    return sampling_methods[sampling_method](datasets, data_size, **kwargs)
