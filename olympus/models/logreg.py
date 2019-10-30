import numpy

from olympus.models.mlp import MLP


def build(input_size, output_size):
    if not isinstance(input_size, int):
        input_size = numpy.product(input_size)

    return MLP(input_size, output_size, layers=[], bias=True)


builders = {'logreg': build}
