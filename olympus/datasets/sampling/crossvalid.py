import numpy

from olympus.datasets.sampling.balanced_classes import balanced_random_indices


def crossvalid_random_indices(rng, indices, n_train, n_valid, n_test):

    indices = numpy.array(indices)

    rng.shuffle(indices)

    train_indices = indices[:n_train]
    valid_indices = indices[n_train:n_train + n_valid]
    test_indices = indices[n_train + n_valid:]

    return dict(train=train_indices, valid=valid_indices, test=test_indices)


def sample(datasets, data_size, seed, **kwargs):
    return balanced_random_indices(
        method=crossvalid_random_indices, classes=datasets.classes, n_points=data_size, seed=seed)
