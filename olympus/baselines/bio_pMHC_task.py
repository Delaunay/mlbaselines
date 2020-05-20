import time
import numpy
import os

import sklearn.neural_network

from olympus.datasets.mhc import get_train_dataset
from olympus.metrics.accuracy import AUC
from olympus.tasks.sklearn_like import SklearnTask
from olympus.observers.msgtracker import metric_logger
from olympus.metrics import NotFittedError



def bootstrap(data, bootstrap_seed):
    rng = numpy.random.RandomState(bootstrap_seed)
    n_train = int(data.shape[0]*0.7)
    n_valid = int(data.shape[0]*0.15)
    n_test = data.shape[0] - n_train - n_valid

    indices = set(range(data.shape[0]))

    train_set = sorted(rng.choice(list(indices), size=n_train, replace=True))

    indices -= set(train_set)

    valid_set = sorted(rng.choice(list(indices), size=n_valid, replace=True))

    indices -= set(valid_set)

    test_set = sorted(rng.choice(list(indices), size=n_test, replace=True))

    return dict(train=(data[train_set], data[train_set,-1]),
                valid=(data[valid_set], data[valid_set,-1]),
                test=(data[test_set], data[test_set,-1]))


class MLPRegressor:
    # We can set/fix hyper
    def __init__(self, random_state, **hyper_parameters):
        self.model_ctor = sklearn.neural_network.MLPRegressor
        self.random_state = random_state
        self.hp = HyperParameters(self.hyperparameter_space(), **hyper_parameters)
        self.model = None

    # List of all the hyper-parameters
    @staticmethod
    def hyperparameter_space():
        return {
            'hidden_layer_sizes': 'uniform(50, 70, discrete=True)',
            'alpha': 'uniform(0, 0.1)'
        }
    # List of hyper-parameters that needs to be set to finish initialization
    def get_space(self):
        return self.hp.missing_parameters()

    # Initialize the model when all the hyper-parameters are there
    def init(self, **hyper_parameters):
        self.hp.add_parameters(**hyper_parameters)
        self.model = self.model_ctor(
            # TODO(Assya): Replace random_state if necessary with what sklearn MLPRegressor
            #              use to set the initial weights.
            random_state=self.random_state,
            **self.hp.parameters(strict=True)
        )

    def predict(self, x):
        try:
            return self.model.predict(x)
        except sklearn.exceptions.NotFittedError as e:
            raise NotFittedError from e

    def fit(self, x, y):
        self.model = self.model.fit(x, y)
        return self.model


def main(bootstrap_seed, model_seed, hidden_layer_sizes=(50,), alpha=0.001,
        allele='HLA-A02:01',
        epoch=0,
        uid=None,
        experiment_name=None,
        client=None):
    """

    Parameters
    ----------
    bootstrap_seed: int
        seed for controling which data-points are selected for training/testing splits
    model_seed: int
        seed for the generation of weights
    hidden_layer_sizes: tuple
        the size of layers ex: (50,) is one layer of 50 neurons
    alpha: float
        L2 penalty (regularization term) parameter.

    """

    # TODO(Assya): Make sure to pass bootstrapping seed and model init seed

    # Load Dataset
    train_data = get_train_dataset(folder='/u/trofimov/simul_hpo/data/pMHC_data',allele=allele)
    dataset_splits = bootstrap(train_data, bootstrap_seed)

    # Compute validation and test accuracy
    additional_metrics = [
        AUC(name='validation', loader=[dataset_splits['valid']]),
        AUC(name='test', loader=[dataset_splits['test']])]

    # Setup the task
    task = SklearnTask(
        MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, solver='lbfgs',
                     alpha=alpha, random_state=model_seed),
        metrics=additional_metrics)

    # Save the result of your experiment inside a db
    if client is not None:
        task.metrics.append(metric_logger(
            client=client,
            experiment=experiment_name))

    hyper_parameters = dict(
        model=dict(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=model_seed,
            alpha=alpha
        )
    )

    show_dict(hyperparameters)

    # initialize the task with you configuration
    task.init(
        uid=uid,
        **hyper_parameters
    )

    # Train
    x, y = dataset_splits['train']
    task.fit(x, y)


    show_dict(task.metrics.value())

    return float(stats['validation_aac'])


if __name__ == '__main__':
    for i in range(100):
        main(model_seed=numpy.random.randint(2**30),
                 bootstrap_seed=numpy.random.randint(2**30),
                 allele='HLA-A02:01')
