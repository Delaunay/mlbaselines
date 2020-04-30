from dataclasses import dataclass, field
from datetime import datetime

import numpy

from sklearn import tree
import sklearn.datasets

from olympus.metrics import MetricList
from olympus.observers import ElapsedRealTime, SampleCount
from olympus.observers.observer import Metric
from olympus.observers.msgtracker import metric_logger
from olympus.tasks.task import Task


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


@dataclass
class Accuracy(Metric):
    model: object = field(default_factory=object)
    accuracies: list = field(default_factory=list)
    name: str = 'validation'
    eval_time: float = 0
    data: list = field(default_factory=list)

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def on_new_epoch(self, task=None, epoch=None, context=None):
        start = datetime.utcnow()
        pred = self.model.predict(self.data[0])

        accuracy = (pred == self.data[1]).mean()

        end = datetime.utcnow()

        self.eval_time = (end - start).total_seconds()
        self.accuracies.append(accuracy)

    def on_start_train(self, task=None, step=None):
        self.on_new_epoch()

    def on_end_train(self, task=None, step=None):
        self.on_new_epoch()

    def value(self):
        if not self.accuracies:
            return {}

        return {
            f'{self.name}_accuracy': float(self.accuracies[-1]),
            f'{self.name}_error_rate': 1 - float(self.accuracies[-1]),
            f'{self.name}_time': self.eval_time
        }


def main(max_depth=None,
         min_samples_split=2,
         min_samples_leaf=1,
         min_weight_fraction_leaf=0,
         random_state=1,
         bootstrap_seed=1,
         epoch=None,
         uid=None,
         experiment_name=None,
         client=None):

    task = Task()
    task.metrics.append(ElapsedRealTime().every(batch=1))
    task.metrics.append(SampleCount().every(batch=1, epoch=1))

    max_depth = int(max_depth) if max_depth is not None else None
    min_samples_split = max(1e-5, min_samples_split)
    min_samples_leaf = int(min_samples_leaf)
    min_weight_fraction_leaf = int(min_weight_fraction_leaf)

    hyper_parameters = {
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'min_weight_fraction_leaf': min_weight_fraction_leaf
        }

    # build the logger
    if client is not None:
        task.metrics.append(metric_logger(client=client, experiment=experiment_name))
        task.metrics.new_trial(hyper_parameters, uid)

    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)

    data = bootstrap(data, target, bootstrap_seed)

    clf = tree.DecisionTreeClassifier(
        criterion='gini', splitter='best', max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=None,
        random_state=random_state
        )

    task.metrics.append(Accuracy(model=clf, data=data['valid'], name='validation'))
    task.metrics.append(Accuracy(model=clf, data=data['test'], name='test'))

    clf = clf.fit(*data['train'])

    # TODO: Log metrics with msglogger
    task.metrics.end_train()

    if client is None:
        print(metrics.value())

    return float(task.metrics.value()['validation_error_rate'])


if __name__ == '__main__':
    for i in range(100):
        print(main(
            random_state=numpy.random.randint(2**30),
            bootstrap_seed=numpy.random.randint(2**30)))
