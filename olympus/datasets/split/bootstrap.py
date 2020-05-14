import numpy

from olympus.utils.log import info
from olympus.datasets.split.balanced_classes import balanced_random_indices, Split


def bootstrap_random_indices(rng, indices, n_train, n_valid, n_test):
    indices = set(indices)
    train_indices = rng.choice(list(indices), size=n_train, replace=True)

    indices -= set(train_indices)

    valid_indices = rng.choice(list(indices), size=n_valid, replace=True)

    indices -= set(valid_indices)

    test_indices = rng.choice(list(indices), size=n_test, replace=True)

    indices -= set(test_indices)

    return Split(train=train_indices, valid=valid_indices, test=test_indices)


def split(datasets, data_size, seed, ratio, index, balanced=True):
    if balanced:
        info('Using balanced bootstrap')
        return balanced_random_indices(
            method=bootstrap_random_indices,
            classes=datasets.classes,
            n_points=data_size, seed=seed,
            split_ratio=ratio)
    else:
        info('Using unbalanced bootstrap')
        n_points = len(datasets)
        n_test = int(numpy.ceil(n_points * ratio))
        n_valid = n_test
        n_train = n_points - n_test - n_valid
        rng = numpy.random.RandomState(int(seed))
        return bootstrap_random_indices(rng, range(n_points), n_train, n_valid, n_test)
