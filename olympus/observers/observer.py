from dataclasses import dataclass
from olympus.utils import warning


@dataclass
class Observer:
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


Metric = Observer
