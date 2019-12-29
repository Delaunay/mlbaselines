import pytest

import torch

from olympus.models import Model, known_models


def test_model_default_hyper():
    m: Model = Model('resnet18', weight_init=None)

    assert dict(m.get_space()) != dict()


def test_model_fixed_init():
    m: Model = Model('resnet18')

    assert dict(m.get_space()) == dict()


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
