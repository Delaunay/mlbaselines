import copy
import os
from shutil import copyfile
from typing import Callable
import hashlib
from datetime import datetime

from olympus.observers.observer import Observer
from olympus.resuming import load_state_dict, state_dict
from olympus.utils import info, warning, set_rng_states, get_rng_states, TimeThrottler
from olympus.utils.functional import flatten
from olympus.utils.options import option
from olympus.utils.storage import BaseStorage, InMemoryStorage
from olympus.observers.earlystopping import IsBest


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


class BadCheckpoint(Exception):
    pass


GLOBAL_BUFFER = InMemoryStorage()


class CheckPointer(Observer):
    """
    Parameters
    ----------
    storage: BaseStorage
        A storage instance that specify where and how to storage task states

    keep_best: str
        Name of the metric to use to keep track of the best weight

    time_buffer: int
        Time in second in between file system writes
        Used to not overwhelm the filesystem.
        Parallel FS can be very slow when dealing with small files like states

    save_init: bool
        Save the initial weights

    Examples
    --------
    Checkpointer that keeps the best states

    >>> from olympus.utils.storage import FileStateStorage
    >>> storage = FileStateStorage(folder='/tmp/chckpt')
    >>> chk = CheckPointer(storage, keep_best='validation_loss')

    Notes
    -----

    The way saving best weights works is, the best weight state is only created if a
    worse weight would overwrite it. That means when the training start no best weights is created
    up until we reach a point in training where the loss start to oscillate, the best weight would be created then.

    When a state is not saved on the file system its state is kept inside the checkpoint as pending, so we do not
    miss the best model state.

    At the end of training, any pending states are pushed to the filesystem

    Because of complex interaction between in memory caching and saving the best weights, time_buffer might not be
    honoured exactly. When the best configuration is found and it should be cached it can happen that the weight
    is forced pushed to the file system instead
    """
    def __init__(self, storage: BaseStorage, keep_best: str = None, time_buffer=option('state.storage.time', 5 * 60, type=int),
                 save_init: bool = False):
        self.storage = storage
        self.frequency_epoch: int = option('checkpoint.frequency_epoch', 1, type=int)

        # Keep best state mechanic
        self.best_name: str = None
        self.keep_best: Callable = None
        if keep_best is not None:
            self.keep_best = IsBest(keep_best)

        self.save_init = save_init

        # Time throttling
        self.time_buffer = time_buffer
        self.last_save = datetime.utcnow()

        # Batch resuming is not supported
        self.frequency_new_trial: int = 1
        self.frequency_end_epoch: int = 1
        # cleanup at the end of training
        self.frequency_end_train: int = 1

        self.epoch: int = 0
        # checkpoint is done last after all other metrics have finished computing their statistics
        self.priority: int = -11
        self.uid = None

        self.pending = None

    def save_pending(self):
        if self.pending is None:
            return False

        is_best, state = self.pending

        name = self.uid
        if self.best_name is not None:
            name = self.best_name

        if is_best:
            self.storage.save(name, state)
            self.pending = None
            return True

    def new_best_name(self):
        return f'{self.keep_best.metric}_{self.keep_best.best}_{self.uid}'

    def save(self, task):
        if self.uid is None:
            raise BadCheckpoint('No uid was given cannot save state')

        was_saved = False
        state = state_dict(task)
        state['rng'] = get_rng_states()

        # Was enough time passed since last save
        now = datetime.utcnow()
        elapsed = now - self.last_save
        should_save = elapsed.total_seconds() > self.time_buffer

        # Is it the best model we have seen so far
        is_best = True
        if self.keep_best is not None:
            is_best = self.keep_best(task.metrics.value())

        if state:
            # Current model is not the best and we did not save the last model in a different path
            # (which is the best right now)
            # So we need to move the last state so it does not get overridden by current state
            if not is_best and self.best_name is None:
                info(f'Saving best ({self.keep_best.metric}: {self.keep_best.best})')
                self.best_name = self.new_best_name()

                was_pending = self.save_pending()
                if not was_pending:
                    self.storage.rename(self.uid, self.best_name)

            if should_save:
                was_saved = self.storage.save(self.uid, state)
                self.save_pending()
                self.pending = None
                self.last_save = datetime.utcnow()
            else:
                self.save_pending()
                self.pending = (is_best, state)

            # we have a new best and the best was saved as with a different filename
            # So we need to change both the best state and the latest state
            if is_best and self.best_name is not None:
                info(f'New best ({self.keep_best.metric}: {self.keep_best.best})')

                self.storage.remove(self.best_name)
                self.best_name = self.new_best_name()

                was_pending = self.save_pending()
                if not was_pending:
                    self.storage.copyfile(self.uid, self.best_name)

        else:
            warning('The state dictionary was empty!')

        if was_saved:
            info('Checkpoint saved')
            return

        info('Skipped Checkpoint')

    def on_end_epoch(self, task, epoch, context):
        self.epoch = epoch
        self.save(task)

    def on_end_train(self, task, step=None):
        if self.pending is None:
            return

        is_best, state = self.pending

        self.storage.save(self.uid, state)
        self.pending = None

    def on_new_trial(self, task, step, parameters, uid):
        """On new trial try to resume the new trial"""
        # Make a unique id for resuming
        self.uid = parameters.get('uid', uid)

        if self.uid is None:
            self.uid = unique_trial_id(task.__class__.__name__, parameters)

        state = self.storage.safe_load(self.uid, device=task.device)

        if state is not None:
            set_rng_states(state['rng'])
            load_state_dict(task, state)
            info(f'Resuming (trial_id: {self.uid})')
        else:
            meta = dict(parameters=parameters, task=type(task).__name__)
            self.storage.save_meta(self.uid, meta)
            info(f'Starting a new (trial_id: {self.uid})')

        if state is None and self.save_init:
            state = state_dict(task)
            # state['rng'] = get_rng_states()
            self.storage.save(f'init_{self.uid}', state)

    def load_best(self, task):
        best = self.best_name
        if self.best_name is None:
            best = self.uid

        state = self.storage.safe_load(best, device=task.device)

        if state is not None:
            set_rng_states(state['rng'])
            load_state_dict(task, state)

    def value(self):
        return {}

    def load_state_dict(self, state_dict, strict=True):
        pass

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {}

