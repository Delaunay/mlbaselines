import numpy

from olympus.datasets.sampling.balanced_classes import balanced_random_indices


def resample_random_indices(rng, indices, n_train, n_valid, n_test, ratio):
    indices = numpy.array(indices)
    n_resampled = int(n_train * ratio)
    rng.shuffle(indices)
    resampled = rng.randint(0, n_resampled, size=n_resampled)
    if len(resampled):
        indices[:n_resampled] = indices[resampled]
    
    train_indices = indices[:n_train]
    valid_indices = indices[n_train:n_train + n_valid]
    test_indices = indices[n_train + n_valid:]

    return dict(train=train_indices, valid=valid_indices, test=test_indices)


def sample(datasets, data_size, seed, ratio, **kwargs):
    return balanced_random_indices(
        method=resample_random_indices, classes=datasets.classes, n_points=data_size, seed=seed, ratio=ratio)
