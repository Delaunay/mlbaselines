try:
    from track import TrackClient, Project, TrialGroup
    from track.client import Trial, TrialDoesNotExist
    NO_TRACK = False

except ImportError:
    NO_TRACK = True


from olympus.utils import info
from olympus.utils.options import options


class BaseTrackLogger:
    def __init__(self, project=None, group=None, storage_uri=options('metric.storage', 'file://track_test.json')):
        pass

    def upsert_trial(self, parameters):
        pass

    def log_metrics(self, step=None, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TrackLogger:
    def __init__(self, project, group, storage_uri=options('metric.storage', 'file://track_test.json')):
        self.client = TrackClient(storage_uri)
        self.client.set_project(Project(name=project))
        self.client.set_group(TrialGroup(name=group))

    def upsert_trial(self, parameters):
        try:
            self.client.set_trial(Trial(parameters=parameters), force=True)
            info('Appending to Track Trial')

        except TrialDoesNotExist:
            self.client.new_trial(force=True, parameters=parameters)

    def log_metrics(self, step=None, **kwargs):
        self.client.log_metrics(step=step, **kwargs)

    def __enter__(self):
        return self.client.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.client.__exit__(exc_type, exc_val, exc_tb)


if NO_TRACK:
    TrackLogger = BaseTrackLogger
