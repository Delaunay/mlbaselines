import numpy

from olympus.datasets.split.resample import resample_random_indices
from olympus.datasets.split.balanced_classes import balanced_random_indices


def subsample_random_indices(rng, indices, n_train, n_valid, n_test, ratio):
    new_indices = resample_random_indices(rng, indices, n_train, n_valid, n_test, ratio)
    new_indices['train'] = numpy.unique(new_indices['train'])
    rng.shuffle(new_indices['train'])
    return new_indices


def split(datasets, data_size, seed, ratio, index):
    return balanced_random_indices(
        method=subsample_random_indices,
        classes=datasets.classes,
        n_points=data_size,
        seed=seed,
        ratio=ratio)
