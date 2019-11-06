from .metric import *
from .adversary import ClassifierAdversary
from .accuracy import OnlineTrainAccuracy, Accuracy
from .progress import ElapsedRealTime, ProgressView, SampleCount


class MetricList:
    """MetricList relays the Event to the Metrics/Observers"""
    def __init__(self, *args):
        self._metrics_mapping = dict()
        self.metrics = list()

        for arg in args:
            self.append(arg)

        self.batch_id: int = 0
        self._epoch: int = 0
        self._previous_step = 0

    def state_dict(self):
        return [m.state_dict() for m in self.metrics]

    def load_state_dict(self, state_dict):
        for m, m_state_dict in zip(self.metrics, state_dict):
            m.load_state_dict(m_state_dict)

    def __getitem__(self, item):
        v = self.get(item)

        if v is None:
            raise RuntimeError('Not found')

    def __setitem__(self, key, value):
        self.append(m=value, key=key)

    def get(self, key, default=None):
        if isinstance(key, int):
            return self.metrics[key]

        if isinstance(key, str):
            return self._metrics_mapping.get(key, default)

        return default

    def append(self, m: Metric, key=None):
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

    def epoch(self, epoch, task=None, context=None):
        for m in self.metrics:
            if m.frequency_epoch > 0 and epoch % m.frequency_epoch == 0:
                m.on_new_epoch(epoch, task, context)

        self._epoch = epoch
        self.batch_id = 0

    def step(self, step, task=None, input=None, context=None):
        # Step back to 0, means it is a new epoch
        if self._previous_step > step:
            assert self.batch_id == 0

        for m in self.metrics:
            if m.frequency_batch > 0 and self.batch_id % m.frequency_batch == 0:
                m.on_new_batch(step, task, input, context)

        self.batch_id += 1
        self._previous_step = step

    def start(self, task=None):
        for m in self.metrics:
            m.start(task)

    def finish(self, task=None):
        for m in self.metrics:
            m.finish(task)

    def value(self):
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.value())

        return metrics

    def report(self, pprint=True, print_fun=print):
        metrics = self.value()

        if pprint:
            print_fun(json.dumps(metrics, indent=2))

        return metrics
