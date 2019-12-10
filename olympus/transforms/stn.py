import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class SpatialTransformerNetwork(nn.Module):
    """Convolutional Neural Networks define an exceptionally powerful class of models,
    but are still limited by the lack of ability to be spatially invariant to the input data in a computationally
    and parameter efficient manner. In this work we introduce a new learnable module, the Spatial Transformer,
    which explicitly allows the spatial manipulation of data within the network.

    More on `arxiv <https://arxiv.org/abs/1506.02025>`_.

    References
    ----------
    .. [1] Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu,
        "Spatial Transformer Networks"
    """
    def __init__(self, input_shape, output_shape=None):
        super(SpatialTransformerNetwork, self).__init__()

        out = torch.randn((1,) + input_shape)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        out = self.localization(out)
        self.size = np.product(out.shape[1:])
        out = out.view(-1, self.size)

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.size, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        _ = self.fc_loc(out)

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.size)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    @staticmethod
    def get_space():
        return {}

