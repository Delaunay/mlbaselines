
import pytest

import torch

from olympus.optimizers import Optimizer
from olympus.optimizers.schedules import LRSchedule, known_schedule
from olympus.models import Model

schedules = known_schedule()


@pytest.mark.parametrize('schedule', schedules)
def test_build_schedule(schedule, batch_size=1):
    model = Model('logreg', weight_init='glorot_uniform', input_size=(1, 28, 28), output_size=(10,))

    optimizer = Optimizer(
        'sgd',
        params=model.parameters()
    )
    optimizer.init(**optimizer.defaults)

    schedule = LRSchedule(
        schedule,
        optimizer=optimizer
    )
    schedule.init(**schedule.defaults)

    optimizer.zero_grad()
    input = torch.randn((batch_size, 1, 28, 28))
    loss = model(input).sum()

    optimizer.backward(loss)
    optimizer.step()

    schedule.step(1)
    schedule.epoch(1)
