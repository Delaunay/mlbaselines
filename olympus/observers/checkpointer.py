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
    # uid = o_params.get('uid')
    # if uid is not None:
    #     return uid

    params = copy.deepcopy(o_params)

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
    uid = None

    def save(self, task):
        was_saved = False
        state = state_dict(task)

        if state:
            was_saved = self.storage.save(self.uid, state)
        else:
            warning('The state dictionary was empty!')

        if was_saved:
            info('Checkpoint saved')
            return

        info('Skipped Checkpoint')

    def on_end_epoch(self, task, epoch, context):
        self.epoch = epoch
        self.save(task)

    def on_new_trial(self, task, step, parameters, uid):
        """On new trial try to resume the new trial"""
        # Make a unique id for resuming
        uid = parameters.get('uid', uid)

        if uid is None:
            self.uid = unique_trial_id(task.__class__.__name__, parameters)

        state = self.storage.safe_load(self.uid, device=task.device)

        if state is not None:
            load_state_dict(task, state)
            info(f'Resuming (trial_id: {self.uid})')
        else:
            info(f'Starting a new (trial_id: {self.uid})')

    def value(self):
        return {}

    def load_state_dict(self, state_dict, strict=True):
        pass

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {}

