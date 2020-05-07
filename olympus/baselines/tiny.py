import numpy

from sklearn import tree
import sklearn.datasets

from olympus.metrics import Accuracy, NotFittedError
from olympus.observers import ElapsedRealTime, SampleCount
from olympus.observers.msgtracker import metric_logger
from olympus.tasks.task import Task
from olympus.utils import HyperParameters, drop_empty_key

from sspace.space import compute_identity


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
class DecisionTree:
    # We can set/fix hyper
    def __init__(self, random_state, **hyper_parameters):
        self.model_ctor = tree.DecisionTreeClassifier
        self.random_state = random_state
        self.hp = HyperParameters(self.hyperparameter_space(), **hyper_parameters)
        self.model = None

    # List of all the hyper-parameters
    @staticmethod
    def hyperparameter_space():
        return {
            'max_depth': 'uniform(0, 100, discrete=True)',
            'min_samples_split': 'uniform(1, 10, discrete=True)',
            'min_samples_leaf': 'uniform(1, 10, discrete=True)',
            'min_weight_fraction_leaf': 'uniform(0, 1)'
        }

    # List of hyper-parameters that needs to be set to finish initialization
    def get_space(self):
        return self.hp.missing_parameters()

    # Initialize the model when all the hyper-parameters are there
    def init(self, **hyper_parameters):
        self.hp.add_parameters(**hyper_parameters)
        self.model = self.model_ctor(
            criterion='gini',
            splitter='best',
            max_features=None,
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


class SklearnTask(Task):
    def __init__(self, model):
        super(SklearnTask, self).__init__()
        self.model = model

        # Measure the time spent training
        self.metrics.append(ElapsedRealTime().every(batch=1))
        self.metrics.append(SampleCount().every(batch=1))

    def get_space(self):
        """Return hyper parameter space of the task"""
        # in that simple case only the model has HP
        # but in DL you would have the optimizer, lr-scheduler, etc...
        return drop_empty_key({
            'model': self.model.get_space(),
        })

    def init(self, model, uid=None):
        self.model.init(**model)

        # Get a unique identifier for this configuration
        if uid is None:
            uid = compute_identity(model, size=16)

        # broadcast a signal that the model is ready
        # so we can setup logging, data, etc...
        self.metrics.new_trial(model, uid)

    def fit(self, x, y, epoch=None, context=None):
        # broadcast to observers/metrics that we are starting the training
        self.metrics.start_train()
        self.metrics.new_epoch(0)
        self.metrics.new_batch(0, input=x)

        self.model.fit(x, y)

        self.metrics.end_batch(1, input=x)
        self.metrics.end_epoch(1)
        # broadcast to observers/metrics that we are ending the training
        self.metrics.end_train()

    def accuracy(self, x, y):
        # How to measure accuracy given our model
        pred = self.model.predict(x)
        accuracy = (pred == y).mean()

        # We expect accuracy and loss
        return accuracy, 0

    # If you support resuming implement those methods
    def load_state_dict(self, state, strict=True):
        pass

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        pass


def main(max_depth=None,
         min_samples_split=2,
         min_samples_leaf=1,
         min_weight_fraction_leaf=0,
         random_state=1,
         bootstrap_seed=1,
         epoch=0,
         uid=None,
         experiment_name=None,
         client=None):

    max_depth = int(max_depth) if max_depth is not None else None
    min_samples_split = max(1e-5, min_samples_split)
    min_samples_leaf = int(min_samples_leaf)
    min_weight_fraction_leaf = int(min_weight_fraction_leaf)

    # Load Dataset
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    dataset_splits = bootstrap(data, target, bootstrap_seed)

    # Setup the task
    task = SklearnTask(DecisionTree(random_state))

    # Compute validation and test accuracy
    task.metrics.append(Accuracy(name='validation', loader=[dataset_splits['valid']]))
    task.metrics.append(Accuracy(name='test', loader=[dataset_splits['test']]))

    # Save the result of your experiment inside a db
    if client is not None:
        task.metrics.append(metric_logger(
            client=client,
            experiment=experiment_name))

    # initialize the task with you configuration
    task.init(
        uid=uid,
        model=dict(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf),
        )

    # Train
    x, y = dataset_splits['train']
    task.fit(x, y)

    # Get the stats about this task setup
    stats = task.metrics.value()
    print(stats)

    return float(stats['validation_error_rate'])


if __name__ == '__main__':
    for i in range(100):
        main(random_state=numpy.random.randint(2**30),
             bootstrap_seed=numpy.random.randint(2**30))
