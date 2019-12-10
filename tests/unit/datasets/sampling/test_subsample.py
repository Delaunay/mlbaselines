import numpy

from olympus.datasets.split.subsample import subsample_random_indices


def test_subsample(base_indices, rng, N_TRAIN, N_VALID, N_TEST):
    subsampled_indices = subsample_random_indices(
        rng, base_indices, N_TRAIN, N_VALID, N_TEST, ratio=1.0)
    assert numpy.unique(subsampled_indices['train']).shape == subsampled_indices['train'].shape
    assert subsampled_indices['train'].shape[0] < N_TRAIN

    rng.seed(1)
    resampled_indices = subsample_random_indices(
        rng, base_indices, N_TRAIN, N_VALID, N_TEST, ratio=1.0)
    assert numpy.unique(resampled_indices['train']).shape == subsampled_indices['train'].shape
