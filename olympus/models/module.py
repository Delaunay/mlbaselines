import torch.nn as nn


class Module(nn.Module):
    """Olympus Module interface to guide new users when doing NAS"""
    def __init__(self, input_size=None, output_size=None):
        super(Module, self).__init__()

    @staticmethod
    def get_space():
        return {}
