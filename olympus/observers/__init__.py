import json

from olympus.observers.observer import Observer
from olympus.observers.progress import *
from olympus.observers.checkpointer import CheckPointer
from olympus.observers.tracking import Tracker


class ObserverList:
    """MetricList relays the Event to the Metrics/Observers"""
    def __init__(self, *args):
        self._metrics_mapping = dict()
        self.metrics = list()

        for arg in args:
            self.append(arg)

        self.batch_id: int = 0
        self.trial_id: int = 0
        self._epoch: int = 0
        self._previous_step = 0

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Save all the children states"""
        return [m.state_dict() for m in self.metrics]

    def load_state_dict(self, state_dict, strict=True):
        """Resume all children metrics using a state_dict"""
        for m, m_state_dict in zip(self.metrics, state_dict):
            m.load_state_dict(m_state_dict)

    def __getitem__(self, item):
        v = self.get(item)

        if v is None:
            raise RuntimeError('Not found')

    def __setitem__(self, key, value):
        self.append(m=value, key=key)

    def get(self, key, default=None):
        """Retrieve a metric from its key

        Parameters
        ----------
        key: Union[str, int]

        default: any
            default object returned if not found
        """
        if isinstance(key, int):
            return self.metrics[key]

        if isinstance(key, str):
            return self._metrics_mapping.get(key, default)

        return default

    def append(self, m: Observer, key=None):
        """Insert a new metric to compute

        Parameters
        ----------
        m: Metric
            new metric to insert

        key: Optional[str]
            optional key used to retrieve the metric
            by default the type name will be used as key
        """
        # Use name attribute as key
        if hasattr(m, 'name') and not key:
            key = m.name

        # Use type name as key
        elif not key:
            key = type(m).__name__

        # only insert if there are no conflicts
        if key not in self._metrics_mapping:
            self._metrics_mapping[key] = m

        self.metrics.append(m)
        self.metrics.sort(key=lambda met: met.priority, reverse=True)

    def on_new_epoch(self, epoch, task=None, context=None):
        """Broadcast a `new_epoch` event to all metrics"""
        for m in self.metrics:
            if m.frequency_epoch > 0 and epoch % m.frequency_epoch == 0:
                m.on_new_epoch(epoch, task, context)

        self._epoch = epoch
        self.batch_id = 0

    def on_new_batch(self, step, task=None, input=None, context=None):
        """Broadcast a `new_batch` event to all metrics"""
        # Step back to 0, means it is a new epoch
        if self._previous_step > step:
            assert self.batch_id == 0, 'This is called when metric is in a bad state!'

        for m in self.metrics:
            if m.frequency_batch > 0 and self.batch_id % m.frequency_batch == 0:
                m.on_new_batch(step, task, input, context)

        self.batch_id += 1
        self._previous_step = step

    def on_new_trial(self, task, parameters, trial_id):
        """Broadcast a `new_trial` event"""

        for m in self.metrics:
            if m.frequency_trial > 0 and self.trial_id % m.frequency_trial == 0:
                m.on_new_trial(task, parameters, trial_id)

        self.trial_id += 1

    def start(self, task=None):
        """Broadcast a `start` event to all metrics"""
        for m in self.metrics:
            m.start(task)

    def finish(self, task=None):
        """Broadcast a `finish` event to all metrics"""
        for m in self.metrics:
            m.finish(task)

    def value(self):
        """Returns a dictionary of all computed metrics"""
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.value())

        return metrics

    def report(self, pprint=True, print_fun=print):
        """Pretty prints all the metrics"""
        metrics = self.value()

        if pprint:
            print_fun(json.dumps(metrics, indent=2))

        return metrics


MetricList = ObserverList
