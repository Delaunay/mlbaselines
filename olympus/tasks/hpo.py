import json
import time

from olympus.tasks.task import Task
from olympus.utils import warning, error, info, show_dict, drop_empty_key
from olympus.hpo import OrionClient

from olympus.utils.functional import flatten


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
                 storage='legacy:pickleddb:test.pkl', max_trials=50, **kwargs):
        super(HPO, self).__init__()
        self.name = experiment_name
        self.fidelities = {}
        self.task_maker = task
        self.client = OrionClient(
            algo, storage, max_trials, **kwargs
        )

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
        space = drop_empty_key(task.get_space(**self.fidelities))

        print('Research Space')
        print('-' * 40)
        print(json.dumps(space, indent=2))

        self.metrics.start_train()

        iterator = self.client.new_experiment(self.name, space, objective)
        for idx, trial in enumerate(iterator):

            new_task = self.task_maker()
            self._set_orion_progress(new_task)
            show_dict(flatten(trial.params))

            params = trial.params
            task_arguments = params.pop('task')

            # FIXME: should not use a trial object, should be an ID
            new_task.init(trial=trial, **params)
            new_task.fit(**task_arguments)

            metrics = new_task.metrics.value()
            val = metrics[objective]

            self.client.report_objective('ValidationErrorRate', 1 - val)

        self.metrics.end_train()
        return self.get_best_trial()

    def get_best_trial(self):
        completed_trials = self.client.fetch_trials_by_status('completed')

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
            progress.orion_handle = self.client

    @property
    def best_trial(self):
        return self.get_best_trial()

    def is_done(self):
        return self.client.is_done

    def is_broken(self):
        return self.client.is_broken

    def wait_done(self, timeout=60, sleep_step=0.1):
        if not self.is_broken() and not self.is_done():
            info('Waiting for other workers to finish')

        total_sleep = 0
        while not self.is_broken() and not self.is_done():
            time.sleep(sleep_step)
            total_sleep += sleep_step

            if timeout is not None and total_sleep > timeout:
                error('Timed out waiting for trials to finish')
                break
