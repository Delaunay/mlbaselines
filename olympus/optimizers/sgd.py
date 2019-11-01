import torch.optim

from olympus.optimizers.base import OptimizerBuilder


class SGD(OptimizerBuilder):
    def build(self, model_parameters, weight_decay, lr, momentum):
        return torch.optim.SGD(
            model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def get_space(self):
        return {'lr': 'loguniform(1e-5, 1)',
                'momentum': 'uniform(0, 1)'}


builders = {'sgd': SGD}
