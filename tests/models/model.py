from olympus.models import Model


def test_model_default_hyper():
    m = Model('resnet18')

    assert m.get_space() == {'weight_init': 'choices(glorot_uniform,glorot_normal,uniform,normal,orthogonal,kinit_uniform,kinit_normal)'}


def test_model_fixed_init():
    m = Model('resnet18', weight_init='glorot_uniform')

    assert m.get_space() == {}



if __name__ == '__main__':
    test_model_fixed_init()
