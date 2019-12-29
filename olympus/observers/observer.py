from dataclasses import dataclass, field
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
    frequency_new_epoch: int = field(default=0)
    frequency_new_batch: int = field(default=0)
    frequency_new_trial: int = field(default=0)
    priority: int = field(default=0)

    def on_new_epoch(self, task, epoch, context):
        """Called at the end of an epoch, before a new epoch starts"""
        pass

    def on_new_batch(self, task, step, input=None, context=None):
        """Called after a batch has been processed"""
        pass

    def on_new_trial(self, task, step, parameters, trial_id):
        """Called after a trial has been processed"""
        pass

    def on_start_train(self, task, step=None):
        """Called on ce the training starts

        Notes
        -----
        You should not rely on this function to initialize your metric as it will
        not be called if the training is resumed from a previous state
        """
        pass

    def on_end_train(self, task, step=None):
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
