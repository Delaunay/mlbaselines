from dataclasses import dataclass, field

from olympus.observers.observer import Observer
from olympus.utils import info, warning
from olympus.utils.options import option

from datetime import datetime


try:
    from track import TrackClient, Project, TrialGroup
    from track.client import Trial, TrialDoesNotExist
    NO_TRACK = False

except ImportError:
    NO_TRACK = True


def get_time_delta(start):
    return (datetime.utcnow() - start).total_seconds()


class BaseTrackLogger:
    def __init__(self, project=None, group=None, storage_uri=option('metric.storage', 'file://track_test.json')):
        pass

    def upsert_trial(self, parameters, trial_id):
        pass

    def log_metrics(self, step=None, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TrackLogger:
    def __init__(self, project, *, group=None, storage_uri=option('metric.storage', 'file://track_test.json')):
        self.client = TrackClient(storage_uri)
        self.client.set_project(Project(name=project))
        # print(self.client.project)
        if group:
            self.client.set_group(TrialGroup(name=group))

    def upsert_trial(self, parameters, trial_id):
        # if UID is set then used the UID
        # this so when we are using orion we use its trial and not create another one
        trial = Trial(parameters=parameters)
        if trial_id is not None:
            try:
                trial = Trial()
                trial.uid = trial_id

            except ValueError:
                # trial_id is not a track id so orion backend and
                # track backend are not compatible, just insert a new trial
                self.upsert_trial(parameters, None)
                warning(f'(trial_id: {trial_id}) is not recognized by track, creating its own trial')
                warning(f'trial id is now (trial_id: {self.client.trial.uid})')
                return

        # Try to use an existing trial
        try:
            self.client.set_trial(trial, force=True)
            info('Appending to Track Trial')

        except TrialDoesNotExist:
            assert not trial_id, 'A trial_id was provided which means the trial should exist'
            self.client.new_trial(force=True, parameters=parameters)

    def log_metrics(self, step=None, **kwargs):
        self.client.log_metrics(step=step, **kwargs)

    def __enter__(self):
        return self.client.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.client.__exit__(exc_type, exc_val, exc_tb)


if NO_TRACK:
    TrackLogger = BaseTrackLogger


@dataclass
class Tracker(Observer):
    logger: TrackLogger = None

    # It NEEDS to be called before a trial is created
    frequency_new_trial: int = 1
    frequency_start_train: int = 1
    frequency_end_train: int = 1

    frequency_new_epoch: int = field(
        default_factory=lambda: option('track.frequency_epoch', 1, type=int))
    frequency_end_batch: int = field(
        default_factory=lambda: option('track.frequency_batch', 0, type=int))

    last_save: datetime = None
    epoch: int = 0
    # tracking is done last after all other metrics have finished computing their statistics
    priority: int = -10

    def on_new_trial(self, task, step, parameters, uid):
        self.logger.upsert_trial(parameters, trial_id=uid)

    # We push data on new epoch so for the last epoch
    # end_train push the last metrics without duplicates
    def on_new_epoch(self, task, epoch, context):
        self.epoch = epoch
        self.logger.log_metrics(step=epoch, **task.metrics.value())

    def on_start_train(self, task, step=None):
        self.logger.__enter__()

    def on_end_train(self, task, step=None):
        if task is not None:
            self.logger.log_metrics(step=0, **task.metrics.value())
        self.logger.__exit__(None, None, None)

    def value(self):
        return {}

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
