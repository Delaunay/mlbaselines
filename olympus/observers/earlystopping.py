from dataclasses import dataclass
from typing import Callable

from olympus.observers.observer import Metric
from olympus.resuming import state_dict, load_state_dict


class IsBest:
    """Returns true if the current model instance is the best we have seen
    According to a given metric

    Parameters
    ----------
    metric_name: str
        Name of the metric observed

    reduce: Callable[[float, float], float]
        How to combine the best metric value with the current one
    """
    def __init__(self, metric_name, reduce=min):
        self.metric = metric_name
        self.reduce = reduce
        self.best = None
        self.best_epoch = None

    def __call__(self, metrics):
        v = metrics.get(self.metric, None)

        if v is not None:
            if self.best is None:
                self.best = v
                return True

            new_best = self.reduce(self.best, v)

            if new_best == self.best:
                return False
            else:
                self.best = new_best
                return True

        return False

    def state_dict(self):
        return dict(
            best_epoch=self.best_epoch,
            metric=self.metric,
            reduce=self.reduce,
            best=self.best)

    def load_state_dict(self, state):
        self.metric = state['metric']
        self.reduce = state['reduce']
        self.best = state['best']
        self.best_epoch = state['best_epoch']


class NoImprovement:
    """Stop training after no improvement was registered for a given metric

    Parameters
    ----------
    window: int
        Number of epochs without improvements

    metric_name: str
        Name of the metric observed

    reduce: Callable[[float, float], float]
        How to combine the best metric value with the current one
    """
    def __init__(self, window, metric_name, reduce=min):
        self.window = window
        self.metric = metric_name
        self.reduce = reduce
        self.best = None
        self.n = 0
        self.is_best = IsBest(metric_name, reduce)

    def __call__(self, metrics):
        is_best = self.is_best(metrics)

        if is_best:
            self.n = 0
        else:
            self.n += 1

        if self.n >= self.window:
            return True

        return False

    def state_dict(self):
        return dict(
            window=self.window,
            metric=self.metric,
            reduce=self.reduce,
            best=self.best,
            n=self.n)

    def load_state_dict(self, state):
        self.window = state['window']
        self.metric = state['metric']
        self.reduce = state['reduce']
        self.best = state['best']
        self.n = state['n']


@dataclass
class EarlyStopping(Metric):
    """Allow to stop training earlier if we meet a given criterion"""
    criterion: Callable = None
    stopped: bool = False
    frequency_end_epoch: int = 1

    priority: int = -9

    def state_dict(self):
        state = dict(stopped=self.stopped)
        state['criterion'] = state_dict(self.criterion)
        return state

    def load_state_dict(self, state_dict):
        self.stopped = state_dict['stopped']
        load_state_dict(self.criterion, state_dict['criterion'])

    def on_end_epoch(self, task, epoch, context):
        if task is not None:
            metrics = task.metrics.value()

            if self.criterion(metrics):
                task.stopped = True

    def value(self):
        return {
            'early_stopped': self.stopped
        }

