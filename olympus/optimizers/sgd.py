import torch.optim

from olympus.optimizers.base import OptimizerBuilder


class SGD(OptimizerBuilder):
    """SGD with momentum, more on `wikipedia <https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum>`_

    References
    ----------
    .. [1] Aleksandar Botev, Guy Lever, David Barber.
        "Nesterov's Accelerated Gradient and Momentum as approximations to Regularised Update Descent", 7 Jul 2016
    """
    def build(self, model_parameters, weight_decay, lr, momentum):
        return torch.optim.SGD(
            model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def get_space(self):
        return {'lr': 'loguniform(1e-5, 1)',
                'momentum': 'uniform(0, 1)'}


builders = {'sgd': SGD}
