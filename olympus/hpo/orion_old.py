from orion.client import create_experiment
from orion.core.worker.trial import Trial

from olympus.distributed.multigpu import rank
from olympus.utils.options import options
from olympus.utils.storage import StateStorage


class OrionClient:
    def __init__(self):
        self.storage = None
        self.experiment = None
        self.trial = None
        self.state_storage = StateStorage(folder=options('state_storage', '/tmp'))
        # TODO: Make this optional for testing...
        self.storage = {
            'type': 'legacy',
            'database': {
                'type': 'pickleddb',
                'name': f'test.pkl'
            }
        }

    def new_trial(self, name, algorithms, space):
        # fetch or create the experiment being ran
        self.experiment = create_experiment(
            name=name,
            storage=self.storage,
            algorithms=algorithms,
            space=space
        )

    def sample(self, args):
        assert self.experiment, 'Experiment needs to be defined first'
        # ---
        # Only the main process can sample the arguments to use
        if rank() == -1:
            self.trial = self.experiment.suggest()

            if self.trial is None:
                raise RuntimeError('Algorithm could not sample a new point')

            for k, v in self.trial.params.items():
                args[k] = v

            return args

        else:  # Fetch the trial we are trying to run now
            # self.trial = self.experiment.fetch_trial(args...)
            pass
        # ---
        return args

    def resume(self, context):
        # assert self.trial, 'Trial needs to be set to be able to resume Trial'

        # new_context = self.state_storage.load(self.trial.id)
        # context.update(new_context)

        return context

    def checkpoint(self, context):
        # Only the main process or master process can report values
        if rank() == -1 or rank() == 0:
            self.state_storage.save(self.trial.id, context)

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
