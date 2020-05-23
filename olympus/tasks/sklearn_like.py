import copy
from collections import defaultdict
from dataclasses import dataclass

import numpy

from sspace.space import compute_identity

from olympus.observers.observer import Metric
from olympus.tasks.task import Task
from olympus.observers import ElapsedRealTime, SampleCount
from olympus.utils import HyperParameters, drop_empty_key
### added by AT to correct import problem
from sklearn.metrics import roc_curve, auc
import numpy as np


class SklearnTask(Task):
    def __init__(self, model, metrics, name=None):
        super(SklearnTask, self).__init__()
        self.model = model

        # Measure the time spent training
        self.metrics.name = name
        self.metrics.append(ElapsedRealTime().every(batch=1))
        self.metrics.append(SampleCount().every(batch=1))
        for metric in metrics:
            self.metrics.append(metric)

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

    def auc(self, x, y):
        # How to measure accuracy given our model
        preds = self.model.predict(x)
        ### added this transformation for ROCAUC, since it requires classes
        y_thresholded = (y>0.5)*1
        fpr, tpr, _  = roc_curve(y_thresholded, preds)
        auc_result = auc(fpr,tpr)

        pcc = np.corrcoef(preds, y)[0, 1]

        return auc_result, pcc

    # If you support resuming implement those methods
    def load_state_dict(self, state, strict=True):
        pass

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        pass


@dataclass
class EnsembleMetric(Metric):

    def __init__(self, task):
        self.task = task

    def _get_keys(self):
        keys = None

        for name, subtask in self.task.tasks.items():
            subtask_keys = set(key[len(name) + 1:] for key in subtask.metrics.value().keys())

            if keys is None:
                keys = subtask_keys
            else:
                keys = keys & subtask_keys

        return keys

    def _get_stats(self):
        keys = self._get_keys()

        metrics = defaultdict(list)

        for name, subtask in self.task.tasks.items():
            values = subtask.metrics.value()
            for key in keys:
                metrics[key].append(values[f'{name}_{key}'])

        stats = dict()
        for key in keys:
            data = numpy.array(metrics[key])
            stats[f'mean_{key}'] = float(data.mean())
            stats[f'std_{key}'] = float(data.std())
            stats[f'min_{key}'] = float(data.min())
            stats[f'max_{key}'] = float(data.max())

        return stats

    def value(self):
        values = {}
        for name, subtask in self.task.tasks.items():
            values.update(subtask.metrics.value())

        values.update(self._get_stats())

        return values


class SklearnEnsembleTask(Task):
    def __init__(self, models, create_subtask_metrics):
        super(SklearnEnsembleTask, self).__init__()
        self.name = ''
        self.models = models

        self.tasks = {}
        for name, model in models.items():
            self.tasks[name] = SklearnTask(model, create_subtask_metrics(name), name=name)

        # Measure the time spent training
        self.metrics.append(ElapsedRealTime().every(batch=1))
        self.metrics.append(EnsembleMetric(self))

    def get_space(self):
        """Return hyper parameter space of the task"""
        # We suppose all models have the same space
        return drop_empty_key({
            'model': self.models[0].get_space(),
        })

    def init(self, model, uid=None):
        for task in self.tasks.values():
            task.init(model=model, uid=uid)

    def fit(self, data, epoch=None, context=None):
        # broadcast to observers/metrics that we are starting the training

        self.metrics.start_train()
        self.metrics.new_epoch(0)
        self.metrics.new_batch(0)

        for name, task in self.tasks.items():
            task.fit(*data[name], context=context)

        self.metrics.end_batch(1)
        self.metrics.end_epoch(1)

        # broadcast to observers/metrics that we are ending the training
        self.metrics.end_train()

    def accuracy(self, x, y):
        accuracies = []

        for name, task in self.tasks.keys():
            for key, value in task.metrics.value():
                if key.endswith('accuracy'):
                    accuracies.append(value)

        # We expect accuracy and loss
        return float(numpy.array(accuracies).mean()), 0

    # If you support resuming implement those methods
    def load_state_dict(self, state, strict=True):
        pass

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        pass

    def get_metrics_value(self):
        # keys = []
        # for name, task in self.tasks.items():
        #     task.metrics.nn

        value = dict()
        for name, task in self.tasks.items():
            value.update(task.metrics.value())

        value.update(self.metrics.value())
        return value
