from olympus.utils import info


def sample(datasets, data_size, **kwargs):
    n_train = datasets.train_size
    n_valid = datasets.valid_size
    n_test = datasets.test_size
    n_points = len(datasets)
    assert n_points == n_train + n_valid + n_test

    info('Using the original split')
    return dict(train=range(n_train), valid=range(n_train, n_train + n_valid),
                test=range(n_train + n_valid, n_points))
