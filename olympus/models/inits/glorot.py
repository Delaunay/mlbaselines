import torch.nn

from olympus.models.inits.base import Initialization


class GlorotUniform(Initialization):
    """
    References
    ----------
    .. [1] Xavier Glorot, Yoshua Bengio,
        "Understanding the difficulty of training deep feedforward neural networks"

    .. [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification", Feb 2015
    """
    def __init__(self, gain=1.0):
        self.gain = gain

    def layer_init(self, weight):
        torch.nn.init.xavier_uniform_(weight, self.gain)


class GlorotNormal(Initialization):
    """See :class`.GlorotUniform`"""
    def __init__(self, gain=1.0):
        self.gain = gain

    def layer_init(self, weight):
        torch.nn.init.xavier_normal_(weight, self.gain)


builders = {
    'glorot_uniform': GlorotUniform,
    'glorot_normal': GlorotNormal}
