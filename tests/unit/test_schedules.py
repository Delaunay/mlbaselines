
import pytest

import torch

from olympus.optimizers import Optimizer
from olympus.optimizers.schedules import LRSchedule, known_schedule
from olympus.models import Model

schedules = known_schedule()


def setup():
    model = Model('logreg', input_size=(28,), output_size=(10,))

    optimizer = Optimizer(
        'sgd',
        params=model.parameters()
    )
    optimizer.init(**optimizer.defaults)
    return model, optimizer


def schedule_work(schedule, optimizer, model):
    optimizer.zero_grad()
    x = torch.randn((3, 28))
    loss = model(x).sum()

    optimizer.backward(loss)
    optimizer.step()

    schedule.step(0)
    schedule.epoch(0)


def test_schedule_full_init():
    model, optimizer = setup()
    schedule = LRSchedule(
        'exponential',
        optimizer=optimizer,
        gamma=0.97
    )
    schedule_work(schedule, optimizer, model)


def test_schedule_lazy_optimizer():
    model, optimizer = setup()
    schedule = LRSchedule('exponential')

    schedule.init(optimizer=optimizer, **schedule.defaults)
    schedule_work(schedule, optimizer, model)


@pytest.mark.parametrize('schedule', schedules)
def test_build_schedule(schedule):
    model, optimizer = setup()

    schedule = LRSchedule(
        schedule,
        optimizer=optimizer
    )
    schedule.init(**schedule.defaults)
    schedule_work(schedule, optimizer, model)
