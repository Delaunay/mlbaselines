import torch.nn

from olympus.models.inits.base import Initialization


class Kaiming(Initialization):
    """
    References
    ----------
    .. [1] Xavier Glorot, Yoshua Bengio,
        "Understanding the difficulty of training deep feedforward neural networks"

    .. [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification", Feb 2015
    """
    def __call__(self, model):
        """Init model using given function for Linear and Conv2d, and {0, 1} for BatchNorm."""
        # TODO: detect_non_linearities and pass relu or leaky_relu according to architecture.
        for m in model.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                self.layer_init(m.weight, 'relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                if m.affine:
                    torch.nn.init.constant_(m.weight, 1.0)
                    torch.nn.init.constant_(m.bias, 0.0)
        return model


class KaimingUniform(Kaiming):
    """See :class`.Kaiming`"""

    def __init__(self, a=0, mode='fan_in'):
        self.a = a
        self.mode = mode

    def layer_init(self, weight, non_linearity='leaky_relu'):
        torch.nn.init.kaiming_uniform_(weight, self.a, self.mode, non_linearity)


class KaimingNormal(Initialization):
    """See :class`.Kaiming`"""

    def __init__(self, a=0, mode='fan_in'):
        self.a = a
        self.mode = mode

    def layer_init(self, weight, non_linearity='leaky_relu'):
        torch.nn.init.kaiming_normal_(weight, self.a, self.mode, non_linearity)


builders = {
    'kinit_uniform': KaimingUniform,
    'kinit_normal': KaimingNormal}

