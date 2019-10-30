from collections import OrderedDict

import numpy


def balanced_random_indices(method, classes, n_points, seed, **kwargs):

    assert n_points % len(classes) == 0, "n_points is not a multiple of number of classes"

    n_points_per_class = n_points // len(classes)
    assert n_points_per_class <= len(classes[0]), "n_points greater than nb of points available"

    n_test_per_class = int(numpy.ceil(n_points_per_class * 0.1))
    n_valid_per_class = n_test_per_class
    n_train_per_class = n_points_per_class - n_test_per_class - n_valid_per_class
    assert n_train_per_class + n_valid_per_class + n_test_per_class == n_points_per_class

    rng = numpy.random.RandomState(int(seed))

    sampled_indices = OrderedDict((
        ('train', []), ('valid', []), ('test', [])))

    for indices in classes:
        class_sampled_indices = method(
            rng, indices, n_train_per_class, n_valid_per_class, n_test_per_class, **kwargs)

        for set_name in sampled_indices.keys():
            sampled_indices[set_name].extend(class_sampled_indices[set_name])

    # Make sure they are not grouped by class
    for set_name in sampled_indices.keys():
        rng.shuffle(sampled_indices[set_name])
        sampled_indices[set_name] = numpy.array(sampled_indices[set_name])

    return sampled_indices
