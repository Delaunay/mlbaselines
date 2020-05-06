from dataclasses import dataclass, field
from datetime import datetime

from olympus.observers.observer import Observer
from olympus.utils.options import option

try:
    from msgqueue.backends import new_client
    ERROR = None
except ImportError as e:
    ERROR = e


METRIC_QUEUE = 'OLYMETRIC'
METRIC_ITEM = 200


def metric_logger(uri=None, database=None, experiment=None, client=None):
    return MSGQTracker(
        client=_Logger(uri=uri, database=database, experiment=experiment, client=client))


class _Logger:
    def __init__(self, uri=None, database=None, experiment=None, client=None):
        if ERROR is not None:
            raise ERROR

        self.experiment = experiment
        if client is None:
            client = new_client(uri, database)
        self.client = client
        self.uid = None

    def log(self, data):
        data['uid'] = self.uid
        self.client.push(METRIC_QUEUE, self.experiment, data, mtype=METRIC_ITEM)


@dataclass
class MSGQTracker(Observer):
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
        assert uid is not None
        self.client.uid = uid

    # We push data on new epoch so for the last epoch
    # end_train push the last metrics without duplicates
    def on_new_epoch(self, task, epoch, context):
        self.epoch = epoch
        self.client.log(task.metrics.value())

    def on_start_train(self, task, step=None):
        pass

    def on_end_train(self, task, step=None):
        if task is not None:
            self.client.log(task.metrics.value())

    def log(self, **kwargs):
        return self.client.log(kwargs)

    def value(self):
        return {}

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
