import numpy

from sklearn import linear_model
import sklearn.datasets

from olympus.metrics import Accuracy, NotFittedError
from olympus.observers import ElapsedRealTime, SampleCount
from olympus.observers.msgtracker import metric_logger
from olympus.tasks.sklearn_like import SklearnTask
from olympus.utils import HyperParameters, drop_empty_key, show_dict


def bootstrap(data, target, seed):
    rng = numpy.random.RandomState(seed)
    n_train = int(data.shape[0]*0.7)
    n_valid = int(data.shape[0]*0.15)
    n_test = data.shape[0] - n_train - n_valid

    indices = set(range(data.shape[0]))

    train_set = sorted(rng.choice(list(indices), size=n_train, replace=True))

    indices -= set(train_set)

    valid_set = sorted(rng.choice(list(indices), size=n_valid, replace=True))

    indices -= set(valid_set)

    test_set = sorted(rng.choice(list(indices), size=n_test, replace=True))

    return dict(train=(data[train_set], target[train_set]),
                valid=(data[valid_set], target[valid_set]),
                test=(data[test_set], target[test_set]))


# The model we want to benchmark
class LogisticRegression:
    # We can set/fix hyper
    def __init__(self, random_state, **hyper_parameters):
        self.model_ctor = linear_model.LogisticRegression
        self.random_state = random_state
        self.hp = HyperParameters(self.hyperparameter_space(), **hyper_parameters)
        self.model = None

    # List of all the hyper-parameters
    @staticmethod
    def hyperparameter_space():
        return {
            'C': 'uniform(0, 1)',
            'l1_ratio': 'uniform(0, 1)'
        }

    # List of hyper-parameters that needs to be set to finish initialization
    def get_space(self):
        return self.hp.missing_parameters()

    # Initialize the model when all the hyper-parameters are there
    def init(self, **hyper_parameters):
        self.hp.add_parameters(**hyper_parameters)
        self.model = self.model_ctor(
            penalty='elasticnet',
            solver='saga',
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


def main(C=1,
         l1_ratio=0.5,
         random_state=1,
         bootstrap_seed=1,
         epoch=0,
         uid=None,
         experiment_name=None,
         client=None):

    C = max(C, 1e-10)

    # Load Dataset
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    dataset_splits = bootstrap(data, target, bootstrap_seed)

    model = LogisticRegression(random_state)

    # Compute validation and test accuracy
    metrics = [
        Accuracy(name='validation', loader=[dataset_splits['valid']]),
        Accuracy(name='test', loader=[dataset_splits['test']])]

    # Setup the task
    task = SklearnTask(model, metrics)

    # Save the result of your experiment inside a db
    if client is not None:
        task.metrics.append(metric_logger(
            client=client,
            experiment=experiment_name))

    hyper_parameters = dict(
        model=dict(
            C=C,
            l1_ratio=l1_ratio
        )
    )

    show_dict(hyper_parameters)

    # initialize the task with you configuration
    task.init(
        uid=uid,
        **hyper_parameters
    )

    # Train
    x, y = dataset_splits['train']
    # TODO: make sure that we fit on whole train and validate on whole valid and test
    task.fit(x, y)

    # Get the stats about this task setup
    stats = task.metrics.value()
    show_dict(stats)

    return float(stats['validation_error_rate'])

# TODO: Adapt conf yaml file
#       Run with searchspace
#       then variance
#       then hpo?
#       then simul 

if __name__ == '__main__':
    main()
