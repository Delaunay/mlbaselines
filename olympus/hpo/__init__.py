from datetime import datetime

# from .orion import OrionClient
from orion.client.experiment import ExperimentClient

from olympus.utils.stat import StatStream


class TrialIterator:
    """Take an Orion experiment and iterate through all the trials it suggests

    Parameters
    ----------
    experiment: ExperimentClient
        Orion Experiment
    """
    def __init__(self, experiment):
        self.experiment = experiment
        self.time = StatStream(drop_first_obs=1)

    def __iter__(self):
        return self

    def __next__(self):
        if self.experiment.is_done or self.experiment.is_broken:
            raise StopIteration

        start = datetime.utcnow()
        trial = self.experiment.suggest()
        self.time += (datetime.utcnow() - start).total_seconds()

        if trial is None:
            raise StopIteration

        return trial

    next = __next__
