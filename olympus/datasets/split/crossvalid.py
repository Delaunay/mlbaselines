import numpy

from olympus.datasets.split.balanced_classes import balanced_random_indices, Split


def crossvalid_random_indices(rng, indices, n_train, n_valid, n_test):

    indices = numpy.array(indices)

    rng.shuffle(indices)

    train_indices = indices[:n_train]
    valid_indices = indices[n_train:n_train + n_valid]
    test_indices = indices[n_train + n_valid:]

    return Split(train=train_indices, valid=valid_indices, test=test_indices)


def split(datasets, data_size, seed, ratio, index):
    return balanced_random_indices(
        method=crossvalid_random_indices,
        classes=datasets.classes,
        n_points=data_size,
        seed=seed)
