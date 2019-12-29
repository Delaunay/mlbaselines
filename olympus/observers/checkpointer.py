import copy
from dataclasses import dataclass, field
import hashlib


from olympus.observers.observer import Observer
from olympus.resuming import load_state_dict, state_dict
from olympus.utils import info, warning
from olympus.utils.functional import flatten
from olympus.utils.options import option
from olympus.utils.storage import BaseStorage


def unique_trial_id(task, o_params):
    params = copy.deepcopy(o_params)
    params.pop('task', None)

    hash = hashlib.sha256()
    hash.update(task.encode('utf8'))

    for k, v in flatten(params).items():
        hash.update(k.encode('utf8'))
        hash.update(str(v).encode('utf8'))

    return hash.hexdigest()


@dataclass
class CheckPointer(Observer):
    storage: BaseStorage = None
    frequency_epoch: int = field(
        default_factory=lambda: option('checkpoint.frequency_epoch', 1, type=int))

    # Batch resuming is not supported
    frequency_new_trial: int = 1
    frequency_end_epoch: int = 1

    epoch: int = 0
    # checkpoint is done last after all other metrics have finished computing their statistics
    priority: int = -11
    trial_id = None

    def save(self, task):
        was_saved = False
        state = state_dict(task)

        if state:
            was_saved = self.storage.save(self.trial_id, state)
        else:
            warning('The state dictionary was empty!')

        if was_saved:
            info('Checkpoint saved')
            return

        info('Skipped Checkpoint')

    def on_end_epoch(self, task, epoch, context):
        self.epoch = epoch
        self.save(task)

    def on_new_trial(self, task, step, parameters, trial):
        """On new trial try to resume the new trial"""
        # Make a unique id for resuming
        if trial is None:
            self.trial_id = unique_trial_id(task.__class__.__name__, parameters)
        else:
            # /!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\
            # /!\ hash_params ignore fidelity     /!\ Do not update this code if you do not know
            # /!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\ what that means
            self.trial_id = trial.hash_params

        state = self.storage.safe_load(self.trial_id, device=task.device)

        if state is not None:
            load_state_dict(task, state)
            info(f'Resuming (trial_id: {self.trial_id})')
        else:
            info(f'Starting a new (trial_id: {self.trial_id})')

    def value(self):
        return {}

    def load_state_dict(self, state_dict, strict=True):
        pass

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {}

