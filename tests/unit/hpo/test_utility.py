from collections import OrderedDict
from olympus.hpo import Fidelity
from olympus.hpo.utility import hyperparameters


@hyperparameters(epoch='fidelity(1, 30, 4)', lr='uniform(0, 1)', b='uniform(0, 1)', c='uniform(0, 1)')
def my_trial(epoch, lr, a, b, c, **kwargs):
    return lr * a - b * c


def test_space():
    assert my_trial.space == OrderedDict([
        ('lr', 'uniform(0, 1)'),
        ('b', 'uniform(0, 1)'),
        ('c', 'uniform(0, 1)')])


def test_fidelity():
    assert my_trial.fidelity == Fidelity(1, 30, 4, name='epoch')

