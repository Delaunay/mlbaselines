from dataclasses import dataclass, field
from olympus.utils import warning



@dataclass
class Metric:
    """Metrics are observers that receives events periodically

    Attributes
    ----------
    frequency_epoch: int
        Controls how often `on_new_epoch` is called, 0 disables it

    frequency_batch: int
        Controls how often `on_new_batch` is called, 0 disables it

    frequency_trial: int
        Controls how often `on_new_trial` is called, 0 disables it

    priority: int
        Controls which metric is called first
    """
    frequency_epoch: int = 0
    frequency_batch: int = 0
    frequency_trial: int = 0
    priority: int = 0

    def on_new_epoch(self, epoch, task, context):
        """Called at the end of an epoch, before a new epoch starts"""
        pass

    def on_new_batch(self, step, task, input, context):
        """Called after a batch has been processed"""
        pass

    def on_new_trial(self, task):
        """Called after a trial has been processed"""
        pass

    def start(self, task=None):
        """Called on ce the training starts

        Notes
        -----
        You should not rely on this function to initialize your metric as it will
        not be called if the training is resumed from a previous state
        """
        pass

    def finish(self, task=None):
        """Called at the end of training after the last epoch"""
        pass

    def value(self):
        """Return the key values that metrics computes"""
        return dict()

    def every(self, *args, epoch=None, batch=None):
        """Define how often this metric should be called"""
        assert len(args) == 0

        if epoch is not None:
            self.frequency_epoch = epoch

        if batch is not None:
            self.frequency_batch = batch

        return self

    def state_dict(self):
        """Return a state dictionary used to checkpointing and resuming"""
        warning(f'This metric {type(self)} does not support resuming')
        return {}

    def load_state_dict(self, state_dict):
        """Load a state dictionary to resume a previous training"""
        warning(f'This metric {type(self)} does not support resuming')


@dataclass
class NamedMetric(Metric):
    """Retrieve a value from the context and track its progress"""
    name: str = None
    metrics: list = field(default_factory=list)
    accumulator = 0
    count = 0

    def on_new_epoch(self, epoch, task, context):
        self.metrics.append(self.accumulator / self.count)
        self.accumulator = 0
        self.count = 0

    def on_new_batch(self, step, task, input, context):
        value = context.get(self.name)
        if value is not None:
            self.accumulator += value
            self.count += 1

    def finish(self, task):
        if self.count > 0:
            self.on_new_epoch(None, None, None)

    def value(self):
        return {
            self.name: self.metrics[-1]
        }
