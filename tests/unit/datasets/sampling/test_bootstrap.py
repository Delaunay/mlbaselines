import numpy

from olympus.datasets.split.bootstrap import bootstrap_random_indices, split


def test_bootstrap_deterministic(base_indices, N_TRAIN, N_VALID, N_TEST):
    indices = bootstrap_random_indices(
        numpy.random.RandomState(1), base_indices, N_TRAIN, N_VALID, N_TEST)
    new_indices = bootstrap_random_indices(
        numpy.random.RandomState(1), base_indices, N_TRAIN, N_VALID, N_TEST)

    for key, _ in new_indices.items():
        assert all(new_indices[key] == indices[key])

    new_indices = bootstrap_random_indices(
        numpy.random.RandomState(2), base_indices, N_TRAIN, N_VALID, N_TEST)

    for key, _ in new_indices.items():
        assert any(new_indices[key] != indices[key])


def test_bootstrap_size(base_indices, rng, N_TRAIN, N_VALID, N_TEST):
    indices = bootstrap_random_indices(rng, base_indices, N_TRAIN, N_VALID, N_TEST)
    assert indices['train'].shape[0] == N_TRAIN
    assert indices['valid'].shape[0] == N_VALID
    assert indices['test'].shape[0] == N_TEST

    assert numpy.unique(indices['train']).shape[0] < N_TRAIN
    assert numpy.unique(indices['valid']).shape[0] < N_VALID
    assert numpy.unique(indices['test']).shape[0] < N_TEST


def test_bootstrap_separation(base_indices, rng, N_TRAIN, N_VALID, N_TEST):
    indices = bootstrap_random_indices(rng, base_indices, N_TRAIN, N_VALID, N_TEST)
    assert set(indices['train']) & set(indices['valid']) == set()
    assert set(indices['train']) & set(indices['test']) == set()
    assert set(indices['valid']) & set(indices['test']) == set()


def test_balanced_bootstrap(N_POINTS, N_CLASSES):
    points_per_class = int(N_POINTS / N_CLASSES)
    classes = [[i * points_per_class + j for j in range(points_per_class)]
               for i in range(N_CLASSES)]
    class Mock():
        def __init__(self, classes):
            self.classes = classes
    indices = split(Mock(classes), N_POINTS, 1, 0.1, index=0, balanced=True)
    assert set(indices['train']) & set(indices['valid']) == set()
    assert set(indices['train']) & set(indices['test']) == set()
    assert set(indices['valid']) & set(indices['test']) == set()


def test_unbalanced_bootstrap(N_POINTS):
    indices = split(list(range(N_POINTS)), None, 1, 0.1, index=0, balanced=False)
    assert set(indices['train']) & set(indices['valid']) == set()
    assert set(indices['train']) & set(indices['test']) == set()
    assert set(indices['valid']) & set(indices['test']) == set()
