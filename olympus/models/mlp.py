import numpy

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, num_classes, layers=tuple(), bias=True):
        self.input_size = input_size
        super(MLP, self).__init__()
        insizes = [input_size] + list(layers)
        outsizes = list(layers) + [num_classes]
        for i, [insize, outsize] in enumerate(zip(insizes, outsizes)):
            setattr(self, 'fc{}'.format(i), nn.Linear(insize, outsize, bias=bias))

    def forward(self, x):
        x = x.view(x.size(0), self.input_size)
        layers = list(self.named_children())
        for name, layer in layers[:-1]:
            x = nn.functional.relu(layer(x))

        return layers[-1][1](x)
