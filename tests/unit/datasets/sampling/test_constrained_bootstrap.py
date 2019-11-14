import numpy

from olympus.datasets.sampling.constrained_bootstrap import constrained_bootstrap_random_indices


def test_constrained_bootstrap_deterministic(base_indices, N_TRAIN, N_VALID, N_TEST):
    indices = constrained_bootstrap_random_indices(
        numpy.random.RandomState(1), base_indices, N_TRAIN, N_VALID, N_TEST)
    new_indices = constrained_bootstrap_random_indices(
        numpy.random.RandomState(1), base_indices, N_TRAIN, N_VALID, N_TEST)

    for key in new_indices:
        assert all(new_indices[key] == indices[key])

    new_indices = constrained_bootstrap_random_indices(
        numpy.random.RandomState(2), base_indices, N_TRAIN, N_VALID, N_TEST)

    for key in new_indices:
        assert any(new_indices[key] != indices[key])


def test_constrained_bootstrap_size(base_indices, rng, N_TRAIN, N_VALID, N_TEST):
    indices = constrained_bootstrap_random_indices(rng, base_indices, N_TRAIN, N_VALID, N_TEST)
    assert indices['train'].shape[0] == N_TRAIN
    assert indices['valid'].shape[0] == N_VALID
    assert indices['test'].shape[0] == N_TEST

    assert numpy.unique(indices['train']).shape[0] < N_TRAIN
    assert numpy.unique(indices['valid']).shape[0] < N_VALID
    assert numpy.unique(indices['test']).shape[0] < N_TEST


def test_constrained_bootstrap_separation(base_indices, rng, N_TRAIN, N_VALID, N_TEST):
    indices = constrained_bootstrap_random_indices(rng, base_indices, N_TRAIN, N_VALID, N_TEST)
    assert set(indices['train']) & set(indices['valid']) == set()
    assert set(indices['train']) & set(indices['test']) == set()
    assert set(indices['valid']) & set(indices['test']) == set()
