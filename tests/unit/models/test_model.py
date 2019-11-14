from olympus.models import Model


def test_model_default_hyper():
    m = Model('resnet18')

    assert m.get_space() != {}


def test_model_fixed_init():
    m = Model('resnet18', weight_init='glorot_uniform')

    assert m.get_space() == {}
