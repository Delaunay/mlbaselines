from olympus.utils.options import option

# Orion Debug
import logging
logging.basicConfig(level=option('orion.debug', logging.DEBUG, type=int))

import_error = None
try:
    from orion.client.experiment import ExperimentClient
    from orion.client import create_experiment
    from orion.core.worker.trial import Trial
except ImportError as e:
    import_error = e

import time
from datetime import datetime

from olympus.distributed.multigpu import rank
from olympus.utils import get_storage, error, info, warning, debug
from olympus.utils.stat import StatStream


class TrialIterator:
    """Take an Orion experiment and iterate through all the trials it suggests

    Parameters
    ----------
    experiment: ExperimentClient
        Orion Experiment
    """
    def __init__(self, experiment, retries=2, client=None):
        self.experiment = experiment
        self.time = StatStream(drop_first_obs=1)
        self.retries = retries
        self.client = client

    def __iter__(self):
        return self

    @property
    def is_finished(self):
        return self.experiment.is_done or self.experiment.is_broken

    def __next__(self, _depth=0):
        from orion.core.worker import WaitingForTrials

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
        try:
            trial = self.experiment.suggest()

        except WaitingForTrials:
            # Do not increase depth this is not a retry
            time.sleep(10)
            return self.next(_depth)

        self.time += (datetime.utcnow() - start).total_seconds()

        if trial is None:
            if not self.experiment.is_done:
                warning('No additional trials were found but experiment is not done')
                return self.next(_depth + 1)
            else:
                info(f'Orion did not suggest more trials (is_done: {self.experiment.is_done}')
                raise StopIteration

        if self.client is not None:
            self.client.trial = trial

        return trial

    next = __next__


class OrionApiClient:
    def __init__(self, algo, storage_uri=option('orion.uri', 'track://file.json'), max_trials=50, **kwargs):
        if import_error is not None:
            raise RuntimeError('Orion is not installed!') from import_error

        self.experiment = None
        self.max_trials = max_trials
        self.storage_uri = storage_uri
        self.trial = None
        self.hpo_config = {
            algo: kwargs
        }

    def new_experiment(self, name, space, objective):
        # fetch or create the experiment being ran
        self.experiment = create_experiment(
            name=name,
            max_trials=self.max_trials,
            space=space,
            algorithms=self.hpo_config,
            strategy='StubParallelStrategy',
            storage=get_storage(self.storage_uri, objective)
        )
        return TrialIterator(self.experiment, client=self)

    def report(self, name, value, type):
        # Only the main process or master process can dashboard values
        if rank() == -1 or rank() == 0:
            assert type in Trial.Result.allowed_types

            self.experiment.observe(
                self.trial,
                [dict(name=name, type=type, value=value)]
            )

    def report_objective(self, name, value):
        if rank() == -1 or rank() == 0:
            return self.report(name, value, type='objective')

    def __getattr__(self, item):
        if hasattr(self.experiment, item):
            return getattr(self.experiment, item)
        raise AttributeError(f'{item} not found')
