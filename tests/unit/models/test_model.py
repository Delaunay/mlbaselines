from olympus.models import Model


def test_model_default_hyper():
    m: Model = Model('resnet18', weight_init=None)

    assert dict(m.get_space()) != dict()


def test_model_fixed_init():
    m: Model = Model('resnet18', weight_init='glorot_uniform')

    assert dict(m.get_space()) == dict()
