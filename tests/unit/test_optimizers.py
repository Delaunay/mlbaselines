import pytest

import torch

from olympus.optimizers import Optimizer, known_optimizers
from olympus.models import Model

optimizers = known_optimizers()


@pytest.mark.parametrize('optimizer', optimizers)
def test_build_optimizer(optimizer, batch_size=1):
    model = Model('logreg', weight_init='glorot_uniform', input_size=(1, 28, 28), output_size=(10,))

    optimizer = Optimizer(
        optimizer,
        params=model.parameters()
    )

    print(optimizer.defaults())
    optimizer.init(**optimizer.defaults())

    optimizer.zero_grad()
    input = torch.randn((batch_size, 1, 28, 28))
    loss = model(input).sum()

    optimizer.backward(loss)
    optimizer.step()


