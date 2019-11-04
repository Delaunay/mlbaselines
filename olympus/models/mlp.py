import numpy

import torch.nn as nn


class MLP(nn.Module):
    """An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer.
    Except for the input nodes, each node is a neuron that uses a nonlinear activation function.
    MLP utilizes a supervised learning technique called backpropagation for training.
    Its multiple layers and non-linear activation distinguish MLP from a linear perceptron.
    It can distinguish data that is not linearly separable.
    More on `wikipedia <https://en.wikipedia.org/wiki/Multilayer_perceptron>`

    Attributes
    ----------
    input_size: Tuple[int, ...]
        Accepted size (any)

    num_classes: int
        Number of output neurons

    layers: List[int]
        Size of hidden layers

    non_linearity: Callable[[tensor], tensor]
        Non linearity or activation function to apply for each layers historically sigmoid or tanh but relu
        is the most popular since it does not have as many numerical problems as the others.

    bias: bool
        Add bias weights to each layers
    """
    def __init__(self, input_size, num_classes, layers=tuple(), non_linearity=nn.functional.relu, bias=True):
        self.input_size = input_size
        super(MLP, self).__init__()

        self.non_linearity = non_linearity
        insizes = [input_size] + list(layers)
        outsizes = list(layers) + [num_classes]
        for i, [insize, outsize] in enumerate(zip(insizes, outsizes)):
            setattr(self, 'fc{}'.format(i), nn.Linear(insize, outsize, bias=bias))

    def forward(self, x):
        x = x.view(x.size(0), self.input_size)
        layers = list(self.named_children())
        for name, layer in layers[:-1]:
            x = self.non_linearity(layer(x))

        return layers[-1][1](x)
