from dataclasses import dataclass, field
from datetime import datetime

from olympus.observers.observer import Observer
from olympus.utils.options import option

from msgqueue.backends import new_client

METRIC_QUEUE = 'OLYMETRIC'


class _Logger:
    def __init__(self, namespace, uri):
        self.client = new_client(uri, namespace)
        self.uid = None

    def log(self, epoch, data):
        self.client.push(METRIC_QUEUE, {
            'epoch': epoch,
            'uid': self.uid,
            'metric': data
        })


@dataclass
class Tracker(Observer):
    client: _Logger = None

    frequency_new_trial: int = 1
    frequency_start_train: int = 1
    frequency_end_train: int = 1

    frequency_new_epoch: int = field(default_factory=lambda: option('track.frequency_epoch', 1, type=int))
    frequency_end_batch: int = field(default_factory=lambda: option('track.frequency_batch', 0, type=int))

    last_save: datetime = None
    epoch: int = 0
    # tracking is done last after all other metrics have finished computing their statistics
    priority: int = -10

    def on_new_trial(self, task, step, parameters, uid):
        self.client.uid = uid

    # We push data on new epoch so for the last epoch
    # end_train push the last metrics without duplicates
    def on_new_epoch(self, task, epoch, context):
        self.epoch = epoch
        self.client.log(epoch, task.metrics.values())

    def on_start_train(self, task, step=None):
        pass

    def on_end_train(self, task, step=None):
        if task is not None:
            self.client.log(self.epoch + 1, task.metrics.values())

    def value(self):
        return {}

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
