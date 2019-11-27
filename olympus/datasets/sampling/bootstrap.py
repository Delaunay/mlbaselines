import numpy

from olympus.datasets.sampling.balanced_classes import balanced_random_indices


def bootstrap_random_indices(rng, indices, n_train, n_valid, n_test):
    indices = set(indices)
    train_indices = rng.choice(list(indices), size=n_train, replace=True)

    indices -= set(train_indices)

    valid_indices = rng.choice(list(indices), size=n_valid, replace=True)

    indices -= set(valid_indices)

    test_indices = rng.choice(list(indices), size=n_test, replace=True)

    indices -= set(test_indices)

    return dict(train=train_indices, valid=valid_indices, test=test_indices)


def sample(datasets, data_size, seed, split_ratio=0.1, **kwargs):
    return balanced_random_indices(
        method=bootstrap_random_indices, classes=datasets.classes, n_points=data_size, seed=seed,
        split_ratio=split_ratio)
