import pytest

import torch

from olympus.models import Model, Initializer


def test_model_fixed_init():
    # Use default init & model is not initialized
    m: Model = Model('resnet18')
    assert dict(m.get_space()) == dict()


def test_model_factory_init():
    # set init using its name
    m: Model = Model(
        'resnet18',
        input_size=(1, 28, 28),
        output_size=10,
        weight_init='normal')

    assert dict(m.get_space()) == dict(initializer=dict(mean='normal(0, 1)', std='normal(1, 1)'))
    m.init(initializer=dict(mean=0, std=1))


def test_model_new_factory_init_fixed():
    # set HP in the constructor
    m: Model = Model(
        'resnet18',
        input_size=(1, 28, 28),
        output_size=10,
        weight_init='normal',
        initializer=dict(mean=0, std=1))
    assert dict(m.get_space()) == {}


def test_model_new_object_init_hp_set():
    init = Initializer('normal')

    # set HP using init
    m: Model = Model(
        'resnet18',
        input_size=(1, 28, 28),
        output_size=10,
        weight_init=init)

    assert dict(m.get_space()) == dict(initializer=dict(mean='normal(0, 1)', std='normal(1, 1)'))
    m.init(initializer=dict(mean=0, std=1))


# models = known_models()
models = [
    'lenet', 'mobilenetv2', 'resnet18', 'logreg'
]


@pytest.mark.parametrize('model', models)
def test_build_model(model, batch_size=1):
    model = Model(model, input_size=(1, 28, 28), output_size=(10,))

    input = torch.randn((batch_size, 1, 28, 28))
    model(input)


if __name__ == '__main__':
    print(models)
