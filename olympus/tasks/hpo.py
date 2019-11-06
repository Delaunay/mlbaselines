from olympus.tasks.task import Task
from olympus.utils import warning
from orion.client import create_experiment
from olympus.utils import get_storage
from olympus.hpo import TrialIterator

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


class HPO(Task):
    """

    Attributes
    ----------
    task: Task
        task to do hyper parameter optimization on
    """
    def __init__(self, task):
        self.task_maker = task
        self.experiment = None
        self._missing_parameters = []

    def run(self, metric='validation_accuracy'):
        """Train the model a few times and return a best trial/set of parameters"""
        task = self.task_maker()
        space = flatten.flatten(task.get_space())
        # task.summary()
        task = None

        self.experiment = create_experiment(
            name='test_123',
            max_trials=50,
            space=space,
            algorithms={
                'asha': {
                    'seed': 1,
                    'num_rungs': 5,
                    'num_brackets': 1
                }},
            storage=get_storage('legacy:pickleddb:test.pkl')
        )

        for trial in TrialIterator(self.experiment):
            new_task = self.task_maker()

            params = flatten.unflatten(trial.params)
            task_arguments = params.pop('task')

            new_task.init(**params)
            new_task.fit(**task_arguments)

            val = new_task.metrics.value()[metric]
            self.experiment.observe(
                trial,
                [dict(name='ValidationErrorRate', value=1 - val, type='objective')])

        best_params = self.experiment.get_trial(uid=self.experiment.stats['best_trials_id'])
        # new_task = self.task_maker()
        return best_params
