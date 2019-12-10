import torch.optim

from olympus.optimizers.base import OptimizerAdapter


class SGD(OptimizerAdapter):
    """SGD with momentum, more on `wikipedia <https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum>`_

    References
    ----------
    .. [1] Aleksandar Botev, Guy Lever, David Barber.
        "Nesterov's Accelerated Gradient and Momentum as approximations to Regularised Update Descent", 7 Jul 2016
    """
    def __init__(self, model_parameters, weight_decay, lr, momentum):
        super(SGD, self).__init__(
            torch.optim.SGD,
            model_parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

    @staticmethod
    def get_space():
        return {
            'lr': 'loguniform(1e-5, 1)',
            'momentum': 'uniform(0, 1)',
            'weight_decay': 'loguniform(1e-10, 1e-3)'
        }

    @staticmethod
    def defaults():
        return {
            'weight_decay': 0.001,
            'lr': 0.001,
            'momentum': 0.9,
        }


builders = {'sgd': SGD}
