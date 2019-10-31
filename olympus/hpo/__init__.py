
from .orion import OrionClient


class TrialIterator:
    def __init__(self, experiment):
        self.experiment = experiment

    def __iter__(self):
        return self

    def __next__(self):
        if self.experiment.is_done or self.experiment.is_broken:
            raise StopIteration

        trial = self.experiment.suggest()

        if trial is None:
            raise StopIteration

        return trial

    next = __next__
