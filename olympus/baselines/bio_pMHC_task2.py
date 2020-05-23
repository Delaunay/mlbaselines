import time
import numpy
import os

import sklearn.neural_network
from olympus.datasets.mhc import get_train_dataset, get_valid_dataset, get_test_dataset
from olympus.metrics.accuracy import AUC
from olympus.tasks.sklearn_like import SklearnTask
from olympus.tasks.sklearn_like import SklearnEnsembleTask
from olympus.observers.msgtracker import metric_logger
from olympus.metrics import NotFittedError
from olympus.utils.options import option
from olympus.utils import HyperParameters, show_dict




def bootstrap(x, rng):

    num_points = x.shape[0]
    indices = set(range(num_points))
    ### moved the rng outside, so that it's the same generator for all 3 datasets
    #rng = numpy.random.RandomState(bootstrap_seed)
    indices = rng.choice(list(indices), size=num_points, replace=True)
    return x[indices]


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
            'solver': 'uniform(0, 3, discrete=True)',
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
        data_path='.',
        epoch=0,
        uid=None,
        experiment_name=None,client=None):
   """

    Parameters
    ----------
    bootstrap_seed: int
        seed for controling which data-points are selected for training/testing splits
    model_seed: int
        seed for the generation of weights
    hidden_layer_sizes: tuple
        the size of layers ex: (50,) is one layer of 50 neurons
    solver: one of {‘lbfgs’, ‘sgd’, ‘adam’}
        solver to use for optimisation
    alpha: float
        L2 penalty (regularization term) parameter.
    ensembling: bool
        decides if yes or no we will use ensembling for the test set

    """
    # Load Dataset
    train_data = get_train_dataset(folder=option('data.path', data_path),task='pan_allele', min_nb_examples=1000)
    valid_data = get_valid_dataset(folder=option('data.path', data_path))
    test_data = get_test_dataset(folder=option('data.path', data_path))

    # one bootstrap seed for all 3 datasets
    rng = numpy.random.RandomState(bootstrap_seed)
    train_data = bootstrap(train_data, rng)
    valid_data = bootstrap(valid_data, rng)
    test_data = bootstrap(test_data, rng)
    
    # Compute validation and test accuracy
    additional_metrics = [
        AUC(name='validation', loader=[([valid_data[:, :-1]], valid_data[:, -1])]),
        AUC(name='test', loader=[([test_data[:, :-1]], test_data[:, -1])])]

    # Setup the task
    task = SklearnTask(
        MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, solver=solver, alpha=alpha),
        metrics=additional_metrics)

    # Save the result of your experiment inside a db
    if client is not None:
        task.metrics.append(metric_logger(
            client=client,
            experiment=experiment_name))

    hyper_parameters = dict(
        model=dict(
            # TODO(Assya) Pass HPs here
        )
    )

    show_dict(hyperparameters)

    # initialize the task with you configuration
    task.init(
        uid=uid,
        **hyper_parameters
    )

    # Train
    task.fit(train_data[:, :-1], train_data[:, -1])

    show_dict(task.metrics.value())

    return float(stats['validation_aac'])

if __name__ == '__main__':
    main(model_seed=numpy.random.randint(2**30),
             bootstrap_seed=numpy.random.randint(2**30))

