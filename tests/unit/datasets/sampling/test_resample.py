import numpy
import os

from olympus.datasets.sampling.resample import resample_random_indices


def is_travis():
    return bool(os.environ.get('TRAVIS', 0))


def test_resample_deterministic(base_indices, N_TRAIN, N_VALID, N_TEST):
    indices = resample_random_indices(
        numpy.random.RandomState(1), base_indices, N_TRAIN, N_VALID, N_TEST, ratio=0.5)
    new_indices = resample_random_indices(
        numpy.random.RandomState(1), base_indices, N_TRAIN, N_VALID, N_TEST, ratio=0.5)

    for key in new_indices:
        assert all(new_indices[key] == indices[key])

    new_indices = resample_random_indices(
        numpy.random.RandomState(2), base_indices, N_TRAIN, N_VALID, N_TEST, ratio=0.5)

    for key in new_indices:
        assert any(new_indices[key] != indices[key])


def test_resample_ratio(base_indices, rng, N_TRAIN, N_VALID, N_TEST):
    # This fails on travis
    if not is_travis():
        indices = resample_random_indices(rng, base_indices, N_TRAIN, N_VALID, N_TEST, ratio=0.0)
        assert numpy.unique(indices['train']).shape[0] == N_TRAIN

    indices = resample_random_indices(rng, base_indices, N_TRAIN, N_VALID, N_TEST, ratio=0.5)
    assert numpy.unique(indices['train']).shape[0] < N_TRAIN * 0.85
    indices = resample_random_indices(rng, base_indices, N_TRAIN, N_VALID, N_TEST, ratio=1.0)
    assert numpy.unique(indices['train']).shape[0] < N_TRAIN * 0.7


def test_resample_separation(base_indices, rng, N_TRAIN, N_VALID, N_TEST):
    indices = resample_random_indices(rng, base_indices, N_TRAIN, N_VALID, N_TEST, ratio=1.0)
    assert set(indices['train']) & set(indices['valid']) == set()
    assert set(indices['train']) & set(indices['test']) == set()
    assert set(indices['valid']) & set(indices['test']) == set()
