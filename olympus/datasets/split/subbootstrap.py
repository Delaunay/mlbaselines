import numpy

from olympus.datasets.split.balanced_classes import balanced_random_indices
from olympus.datasets.split.bootstrap import bootstrap_random_indices


def subbootstrap_random_indices(rng, indices, n_train, n_valid, n_test):

    indices = bootstrap_random_indices(rng, indices, n_train, n_valid, n_test)

    for set_name in ['train', 'valid', 'test']:
        indices[set_name] = numpy.array(list(set(indices[set_name])))

    return indices


def split(datasets, data_size, seed, ratio, index):
    return balanced_random_indices(
        method=subbootstrap_random_indices,
        classes=datasets.classes,
        n_points=data_size,
        seed=seed,
        split_ratio=ratio)
