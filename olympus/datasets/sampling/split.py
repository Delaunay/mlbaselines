import copy

import numpy

from olympus.datasets.sampling.balanced_classes import balanced_random_indices


def split_random_indices(rng, indices, n_train, n_valid, n_test, index):

    indices = numpy.array(copy.deepcopy(indices))
    rng.shuffle(indices)

    n_points = n_train + n_valid + n_test
    start = index * n_points
    if start + n_points > len(indices):
        raise ValueError(
            'Cannot have index `{}` for dataset of size `{}`'.format(
                index, len(indices)))
    train_indices = indices[start:start + n_train]
    valid_indices = indices[start + n_train:start + n_train + n_valid]
    test_indices = indices[start + n_train + n_valid:start + n_train + n_valid + n_test]

    return dict(train=train_indices, valid=valid_indices, test=test_indices)


def sample(datasets, data_size, seed, index, **kwargs):
    return balanced_random_indices(
        method=split_random_indices, classes=datasets.classes, n_points=data_size, seed=seed, index=index)
