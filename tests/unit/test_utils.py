from olympus.utils import HyperParameters, MissingParameters
import pytest


def test_hyperparameter_tracking():
    space = {
        'a': 'uniform(0, 1)',
        'b': 'uniform(0, 1)'
    }

    # space with Fixed HP
    hp = HyperParameters(space, b=0.124)

    # Hp a is missing
    with pytest.raises(MissingParameters):
        hp.parameters(strict=True)

    # return the space of missing params
    assert hp.missing_parameters() == dict(a='uniform(0, 1)')

    hp.add_parameters(a=0.123)
    assert hp.missing_parameters() == {}
    assert hp.parameters(strict=True) == dict(a=0.123, b=0.124)


def test_hyperparameter_nested_tracking():
    space = {
        'initializer': {
            'a': 'uniform(0, 1)',
            'b': 'uniform(0, 1)',
        }
    }

    hp = HyperParameters(space, initializer=dict(b=0.124))

    # Hp a is missing
    with pytest.raises(MissingParameters):
        hp.parameters(strict=True)

    # return the space of missing params
    assert hp.missing_parameters() == dict(initializer=dict(a='uniform(0, 1)'))

    hp.add_parameters(initializer=dict(a=0.123))
    assert hp.missing_parameters() == {}
    assert hp.parameters(strict=True) == dict(initializer=dict(a=0.123, b=0.124))


def test_hyperparameter_nested_tracking_all_set():
    space = {
        'initializer': {
            'a': 'uniform(0, 1)',
            'b': 'uniform(0, 1)',
        }
    }

    hp = HyperParameters(space, initializer=dict(a=0.123, b=0.124))
    assert hp.parameters(strict=True) == dict(initializer=dict(a=0.123, b=0.124))

