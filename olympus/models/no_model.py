import torch
import torch.nn as nn


class NoModel(nn.Module):

    def __init__(self):
        super(NoModel, self).__init__()

    def forward(self, x):
        return x

    def parameters(self, recurse: bool = True):
        return [torch.Tensor([0])]


builders = {'no_model': NoModel}
