from dataclasses import dataclass, field

from olympus.observers.observer import Observer
from olympus.utils.tracker import TrackLogger
from olympus.utils.options import option

from datetime import datetime


def get_time_delta(start):
    return (datetime.utcnow() - start).total_seconds()


@dataclass
class Tracker(Observer):
    logger: TrackLogger = None

    # It NEEDS to be called before a trial is created
    frequency_trial: int = 1
    frequency_epoch: int = field(
        default_factory=lambda: option('track.frequency_epoch', 1, type=int))
    frequency_batch: int = field(
        default_factory=lambda: option('track.frequency_batch', 0, type=int))

    last_save: datetime = None
    epoch: int = 0
    # tracking is done last after all other metrics have finished computing their statistics
    priority: int = -10

    def on_new_trial(self, task, parameters, trial):
        trial_id = None
        if trial is not None:
            trial_id = trial.id

        self.logger.upsert_trial(parameters, trial_id=trial_id)

    def on_new_batch(self, step, task, input, context):
        pass

    def on_new_epoch(self, epoch, task, context):
        self.epoch = epoch
        self.logger.log_metrics(step=epoch, **task.metrics.value())

    def start(self, task=None):
        self.logger.__enter__()
        self.logger.log_metrics(step=0, **task.metrics.value())

    def finish(self, task=None):
        if task is not None:
            self.logger.log_metrics(step=0, **task.metrics.value())
        self.logger.__exit__(None, None, None)

    def value(self):
        return {}

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
