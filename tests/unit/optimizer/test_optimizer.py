import pytest

import torch

from olympus.models import Model
from olympus.optimizers import Optimizer, known_optimizers
from olympus.optimizers.sgd import SGD

optimizers = known_optimizers()


def new_model():
    return Model('logreg', input_size=(28,), output_size=10)


def optimizer_work(optimizer, model):
    x = torch.randn((3, 28))
    optimizer.zero_grad()
    loss = model(x).sum()
    loss.backward()
    optimizer.step()


def test_optimizer_factory_init():
    # set init using its name
    model: Model = new_model()
    optim = Optimizer('sgd', params=model.parameters())

    assert optim.get_space() == SGD.get_space()
    optim.init(**SGD.defaults())


def test_optimizer_factory_init_lazy_model():
    # set init using its name
    model: Model = new_model()
    optim = Optimizer('sgd')

    assert optim.get_space() == SGD.get_space()
    optim.init(params=model.parameters(), **SGD.defaults())

    optimizer_work(optim, model)


def test_optimizer_new_factory_init_fixed():
    # set HP in the constructor
    model: Model = new_model()
    optim = Optimizer('sgd', params=model.parameters(), **SGD.defaults())

    optimizer_work(optim, model)


@pytest.mark.parametrize('optimizer', optimizers)
def test_build_optimizer(optimizer):
    model = new_model()

    optimizer = Optimizer(
        optimizer,
        params=model.parameters()
    )

    optimizer.init(**optimizer.defaults)
    optimizer_work(optimizer, model)
