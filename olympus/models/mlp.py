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

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), self.input_size)
        layers = list(self.named_children())
        for name, layer in layers[:-1]:
            x = nn.functional.relu(layer(x))

        return layers[-1][1](x)
