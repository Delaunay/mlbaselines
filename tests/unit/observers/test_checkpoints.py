import random

import numpy
import pytest
import torch

from olympus.utils import set_verbose_level
set_verbose_level(100)

from olympus.observers import CheckPointer
from olympus.observers.checkpointer import BadCheckpoint
from olympus.resuming import state_dict, load_state_dict
from olympus.utils.storage import StateStorage


class TaskMock:
    def __init__(self, loss=10):
        self.state = 1
        self.loss = loss

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = state_dict(self, destination, prefix, keep_vars, force_default=True)
        state['state'] = self.state
        state['loss'] = self.loss
        return state

    def load_state_dict(self, state):
        load_state_dict(self, state, strict=True, force_default=True)
        self.state = state['state']
        self.loss = state['loss']

    def device(self):
        pass

    @property
    def metrics(self):
        loss = self.loss
        class Dum:
            def value(self):
                return dict(loss=loss)
        return Dum()


def test_checkpoint_bad_setup():
    chk = CheckPointer(storage=StateStorage(folder='/tmp'), time_buffer=0)

    task = TaskMock()
    task.state = 2

    with pytest.raises(BadCheckpoint):
        chk.on_end_epoch(task, 1, dict())

    assert task.state == 2


def test_checkpoint_no_resume():
    chk = CheckPointer(storage=StateStorage(folder='/tmp'), time_buffer=0)

    task = TaskMock()
    task.state = 3
    chk.on_new_trial(task, 0, dict(a=1, b=3), uid='1234')
    chk.on_end_epoch(task, 1, dict())

    task = TaskMock()
    chk.on_new_trial(task, 0, dict(a=3, b=3), uid='fewfef')
    assert task.state == 1


def test_checkpoint_argument():
    chk = CheckPointer(storage=StateStorage(folder='/tmp'), time_buffer=0)

    task = TaskMock()
    task.state = 3
    chk.on_new_trial(task, 0, dict(a=1, b=3), uid='1236')
    chk.on_end_epoch(task, 1, dict())

    task = TaskMock()
    chk.on_new_trial(task, 0, dict(a=1, b=3), uid='1236')
    assert task.state == 3


def test_checkpoint_params():
    chk = CheckPointer(storage=StateStorage(folder='/tmp'), time_buffer=0)

    task = TaskMock()
    task.state = 4
    chk.on_new_trial(task, 0, dict(a=1, b=1, uid='1235'), None)
    chk.on_end_epoch(task, 1, dict())

    task = TaskMock()
    chk.on_new_trial(task, 0, dict(a=1, b=1, uid='1235'), None)
    assert task.state == 4


def test_checkpoint_none():
    chk = CheckPointer(storage=StateStorage(folder='/tmp'), time_buffer=0)

    task = TaskMock()
    task.state = 5
    chk.on_new_trial(task, 0, dict(a=2, b=2), None)
    chk.on_end_epoch(task, 1, dict())

    task = TaskMock()
    chk.on_new_trial(task, 0, dict(a=2, b=2), None)
    assert task.state == 5

    task = TaskMock()
    chk.on_new_trial(task, 0, dict(a=2, b=4), None)
    assert task.state == 1


def test_checkpoint_best():
    s = StateStorage(folder='/tmp/chk')
    chk = CheckPointer(storage=s, time_buffer=0, keep_best='loss')

    task = TaskMock(loss=10000)
    task.loss = 10000
    chk.on_new_trial(task, 0, dict(a=2, b=2), None)
    # Save checkpoint
    chk.on_end_epoch(task, 1, dict())

    ntask = TaskMock()
    chk.load_best(ntask)
    assert ntask.loss == 10000

    #
    task.loss = 1000
    chk.on_end_epoch(task, 2, dict())

    ntask = TaskMock()
    chk.load_best(ntask)
    assert ntask.loss == 1000

    # loss increased
    task.loss = 1001
    chk.on_end_epoch(task, 3, dict())

    ntask = TaskMock()
    chk.load_best(ntask)
    assert ntask.loss == 1000

    task.loss = 999
    chk.on_end_epoch(task, 3, dict())

    ntask = TaskMock()
    chk.load_best(ntask)
    assert ntask.loss == 999

    task.loss = 998
    chk.on_end_epoch(task, 3, dict())

    ntask = TaskMock()
    chk.load_best(ntask)
    assert ntask.loss == 998


def test_checkpoint_rng():
    def get_samples():
        if torch.cuda.is_available():
            a = torch.cuda.FloatTensor(1).normal_()
        else:
            a = 0

        b = random.random()
        c = numpy.random.uniform()
        d = torch.rand(1)
        return a, b, c, d

    chk = CheckPointer(storage=StateStorage(folder='/tmp'), time_buffer=0)

    task = TaskMock()
    task.state = 5
    chk.on_new_trial(task, 0, dict(a=2, b=2), None)
    # Save checkpoint
    chk.on_end_epoch(task, 1, dict())

    a, b, c, d = get_samples()

    a2, b2, c2, d2 = get_samples()

    if torch.cuda.is_available():
        assert a != a2
    assert b != b2
    assert c != c2
    assert d != d2

    task = TaskMock()
    chk.on_new_trial(task, 0, dict(a=2, b=2), None)
    assert task.state == 5

    a2, b2, c2, d2 = get_samples()

    if torch.cuda.is_available():
        assert a == a2
    assert b == b2
    assert c == c2
    assert d == d2
