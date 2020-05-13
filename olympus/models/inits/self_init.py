import torch.nn

from olympus.models.inits.base import Initialization


class SelfInit(Initialization):
    def __init__(self, gain):
        self.gain = gain

    def __call__(self, model):
        model.initialize(gain=self.gain)
        return model

    @staticmethod
    def get_space():
        return {'gain': 'uniform(0, 1)'}


builders = {
    'self_init': SelfInit}
