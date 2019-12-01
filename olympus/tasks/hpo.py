import json
import os
import hashlib

from olympus.tasks.task import Task
from olympus.utils import warning
from olympus.utils import get_storage, show_dict
from olympus.utils.options import options
from olympus.hpo import TrialIterator

from orion.client import create_experiment
from orion.core.utils import flatten


def _generate_arguments(obj_space, all_args):
    """Read the target space and generate a list of arguments for it"""
    init_args = {}

    for k, _ in obj_space.items():
        v = all_args.get(k)
        init_args[k] = v

        if v is None:
            warning(f'hyper-parameter (key: {k}) is missing')

    return init_args


def fidelity(max, min=1, log_base=4):
    return f'fidelity({min}, {max}, {log_base})'


class HPO(Task):
    """

    Attributes
    ----------
    task: Task
        task to do hyper parameter optimization on
    """
    def __init__(self, experiment_name, task, algo,
                 storage='legacy:pickleddb:test.pkl', max_trials=50, folder=options('state.storage', '/tmp'), **kwargs):
        super(HPO, self).__init__()
        self.experiment_name = experiment_name
        self.task_maker = task
        self.experiment = None
        self._missing_parameters = []
        self.fidelities = {}
        self.max_trials = max_trials
        self.folder = folder
        self.storage_uri = storage
        self.hpo_config = {
            algo: kwargs
        }

    @staticmethod
    def _drop_empty_group(space):
        new_space = {}
        for key, val in space.items():
            if val:
                new_space[key] = val

        return new_space

    @staticmethod
    def unique_trial_id(trial, experiment):
        params = trial.params
        # task has the fidelities
        params.pop('task')

        hash = hashlib.sha256()
        hash.update(experiment.encode('utf8'))
        for k, v in flatten.flatten(params).items():
            hash.update(k.encode('utf8'))
            hash.update(str(v).encode('utf8'))

        return hash.hexdigest()

    def fit(self, objective, step=None, input=None, context=None, **fidelities):
        """Train the model a few times and return a best trial/set of parameters"""
        self.fidelities = fidelities

        # >>> import orion.algo.base
        # >>> from orion.algo.asha import compute_budgets
        # >>> compute_budgets(1, 300, reduction_factor=4, num_rungs=4)
        # [1, 6, 44, 300]
        # >>> compute_budgets(1, 300, 4, 5)
        # [1, 4, 17, 72, 300]
        #  1   trial => 300 Epoch
        #  4   trial =>  72
        #  16  trial =>  17
        #  64  trial =>   4
        #  256 trial =>   1

        task = self.task_maker()

        space = HPO._drop_empty_group(task.get_space(**self.fidelities))

        print('Research Space')
        print('-' * 40)
        print(json.dumps(space, indent=2))

        task_name = type(task).__name__.lower()
        experiment_folder = os.path.join(self.folder, task_name, self.experiment_name)

        # task.summary()
        # force early Garbage collect
        task = None
        self.experiment = create_experiment(
            name=self.experiment_name,
            max_trials=self.max_trials,
            space=space,
            algorithms=self.hpo_config,
            storage=get_storage(self.storage_uri, objective)
        )

        self.metrics.start(self)
        iterator = TrialIterator(self.experiment)
        for idx, trial in enumerate(iterator):
            new_task = self.task_maker()
            self._set_orion_progress(new_task)

            # Get a unique ID for the trial checkpointing
            trial_id = HPO.unique_trial_id(trial, experiment_folder)
            new_task.storage.folder = os.path.join(experiment_folder, trial_id)

            show_dict(flatten.flatten(trial.params))

            params = trial.params
            task_arguments = params.pop('task')

            new_task.init(trial_id=trial.id, **params)
            new_task.fit(**task_arguments)

            metrics = new_task.metrics.value()
            val = metrics[objective]

            results = [dict(name='ValidationErrorRate', value=1 - val, type='objective')]
            for k, v in metrics.items():
                results.append(dict(name=k, value=v, type='statistic'))

            self.experiment.observe(trial, results)

        self.metrics.finish(self)
        return self.get_best_trial()

    def get_best_trial(self):
        completed_trials = self.experiment.fetch_trials_by_status('completed')

        best_eval = completed_trials[0].objective.value
        best_trial = completed_trials[0]

        for trial in completed_trials:
            objective = trial.objective.value

            if objective < best_eval:
                best_eval = objective
                best_trial = trial

        return best_trial

    def _set_orion_progress(self, task):
        progress = task.metrics.get('ProgressView')
        if progress:
            progress.orion_handle = self.experiment

    @property
    def best_trial(self):
        return self.get_best_trial()
