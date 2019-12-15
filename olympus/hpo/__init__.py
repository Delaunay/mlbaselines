from datetime import datetime

from olympus.distributed.multigpu import rank
from olympus.utils import get_storage, error, info, warning, debug
from olympus.utils.options import option
from olympus.utils.stat import StatStream

# Orion Debug
import logging
logging.basicConfig(level=option('orion.debug', logging.WARN, type=int))

from orion.client.experiment import ExperimentClient
from orion.client import create_experiment
from orion.core.worker.trial import Trial


class TrialIterator:
    """Take an Orion experiment and iterate through all the trials it suggests

    Parameters
    ----------
    experiment: ExperimentClient
        Orion Experiment
    """
    def __init__(self, experiment, retries=2):
        self.experiment = experiment
        self.time = StatStream(drop_first_obs=1)
        self.retries = retries

    def __iter__(self):
        return self

    @property
    def is_finished(self):
        return self.experiment.is_done or self.experiment.is_broken

    def __next__(self, _depth=0):
        if _depth >= self.retries:
            debug(f'Retried {_depth} times without success')
            raise StopIteration

        if self.experiment.is_broken:
            error('Experiment is broken and cannot continue')
            raise StopIteration

        if self.experiment.is_done:
            info('Orion does not have more trials to suggest')
            raise StopIteration

        start = datetime.utcnow()
        trial = self.experiment.suggest()
        self.time += (datetime.utcnow() - start).total_seconds()

        if trial is None:
            if not self.experiment.is_done:
                warning('No additional trials were found but experiment is not done')
                return self.__next__(_depth + 1)
            else:
                info(f'Orion did not suggest more trials (is_done: {self.experiment.is_done}')
                raise StopIteration

        return trial

    next = __next__


class OrionClient:
    def __init__(self, storage_uri=option('orion.uri', 'track://file.json')):
        self.experiment = None
        self.trial = None
        self.storage = get_storage(storage_uri)

    def new_experiment(self, name, algorithms, space, objective):
        # fetch or create the experiment being ran
        self.experiment = create_experiment(
            name=name,
            algorithms=algorithms,
            space=space,
            storage=self.storage + f'?objective={objective}'
        )

    def __iter__(self):
        return TrialIterator(self.experiment)

    def report(self, name, value, type):
        # Only the main process or master process can report values
        if rank() == -1 or rank() == 0:
            assert type in Trial.Result.allowed_types

            self.experiment.observe(
                self.trial,
                [dict(name=name, type=type, value=value)]
            )

    def report_objective(self, name, value):
        return self.report(name, value, type='objective')
