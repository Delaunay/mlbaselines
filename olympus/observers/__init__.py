import json

from olympus.observers.observer import Observer
from olympus.observers.progress import *
from olympus.observers.checkpointer import CheckPointer
from olympus.observers.msgtracker import metric_logger
from olympus.utils import warning, error


new_trial   = 'new_trial'       # HPO created a trial
start_train = 'start_train'     # Train is starting from scratch
resume_train= 'resume_train'    # Train is being resumed
new_epoch   = 'new_epoch'       # New epoch is starting
end_epoch   = 'end_epoch'
new_batch   = 'new_batch'       # New Batch is starting
end_batch   = 'end_batch'
end_train   = 'end_train'       # Train has finished


class ObserverList:
    """MetricList relays the Event to the Metrics/Observers"""
    def __init__(self, *args, task=None):
        self._metrics_mapping = dict()
        self.metrics = list()

        for arg in args:
            self.append(arg)

        self.batch_id: int = 0
        self.trial_id: int = 0
        self._epoch: int = 0
        self._previous_step = 0
        self.task = task

    @staticmethod
    def should_run(metric, name, step):
        if step is None:
            warning(f'step is none; cannot run (metric: {metric}) with (event: {name})')
            return False

        frequency = getattr(metric, f'frequency_{name}', 1)

        if frequency > 0:
            return step % frequency == 0

        return False

    def broadcast_event(self, event_name, task, step, *args, **kwargs):
        for m in self.metrics:

            if ObserverList.should_run(m, event_name, step):
                fun = getattr(m, f'on_{event_name}', None)

                if fun is not None:
                    try:
                        fun(task, step, *args, **kwargs)
                    except TypeError:
                        error(f'(metric: {m}) (event: {event_name})')
                        raise

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

    def new_epoch(self, epoch, context=None):
        """Broadcast a `new_epoch` event to all metrics"""
        self.broadcast_event('new_epoch', self.task, epoch, context)

    def new_batch(self, step, input=None, context=None):
        """Broadcast a `new_batch` event to all metrics"""
        self.broadcast_event('new_batch', self.task, step, input, context)

    def end_epoch(self, epoch, context=None):
        """Broadcast a `new_epoch` event to all metrics"""
        self.broadcast_event('end_epoch', self.task, epoch, context)

    def end_batch(self, step, input=None, context=None):
        """Broadcast a `new_batch` event to all metrics"""
        self.broadcast_event('end_batch', self.task, step, input, context)

    def new_trial(self, parameters, uid):
        """Broadcast a `new_trial` event"""
        self.broadcast_event('new_trial', self.task, 0, parameters, uid)

    def start_train(self):
        """Broadcast a `start` event to all metrics"""
        self.broadcast_event('start_train', self.task, 0)

    def resume_train(self, start_epoch):
        """Broadcast a `resume` event to all metrics"""
        self.broadcast_event('resume_train', self.task, start_epoch)

    def end_train(self):
        """Broadcast a `finish` event to all metrics"""
        self.broadcast_event('end_train', self.task, 0)

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
