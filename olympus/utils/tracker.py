try:
    from track import TrackClient, Project, TrialGroup
    from track.client import Trial, TrialDoesNotExist
    NO_TRACK = False

except ImportError:
    NO_TRACK = True


from olympus.utils import info, warning
from olympus.utils.options import options


class BaseTrackLogger:
    def __init__(self, project=None, group=None, storage_uri=options('metric.storage', 'file://track_test.json')):
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
    def __init__(self, project, *, group=None, storage_uri=options('metric.storage', 'file://track_test.json')):
        self.client = TrackClient(storage_uri)
        self.client.set_project(Project(name=project))
        if group:
            self.client.set_group(TrialGroup(name=group))

    def upsert_trial(self, parameters, trial_id):
        # if UID is set then used the UID
        # this so when we are using orion we use its trial and not create another one
        trial = Trial(parameters=parameters)
        if trial_id:
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
