import numpy

from olympus.datasets.sampling.resample import resample_random_indices
from olympus.datasets.sampling.balanced_classes import balanced_random_indices


def subsample_random_indices(rng, indices, n_train, n_valid, n_test, ratio):
    new_indices = resample_random_indices(rng, indices, n_train, n_valid, n_test, ratio)
    new_indices['train'] = numpy.unique(new_indices['train'])
    rng.shuffle(new_indices['train'])
    return new_indices


def sample(datasets, data_size, seed, ratio, **kwargs):
    return balanced_random_indices(
        method=subsample_random_indices, classes=datasets.classes, n_points=data_size, seed=seed,
        ratio=ratio)
