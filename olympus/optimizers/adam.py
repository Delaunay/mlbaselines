import torch.optim

from olympus.optimizers.base import OptimizerAdapter


class Adam(OptimizerAdapter):
    """Adam (Adaptive Moment estimation),
    an algorithm for first-order gradient-based optimization of stochastic objective functions,
    based on adaptive estimates of lower-order moments. The method is straightforward to implement,
    is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients,
    and is well suited for problems that are large in terms of data and/or parameters.
    The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients.
    The hyper-parameters have intuitive interpretations and typically require little tuning.
    More on `arxiv <https://arxiv.org/abs/1412.6980>`_

    See also :class`.AMSGrad`

    Attributes
    ----------
    model_parameters: List[Tensor]

    weight_decay: float
        Add L2 penalty to the cost (encourage smaller weights)

    learning_rate: float = 0.001

    beta1: float ∈ [0, 1) default =  0.9
        Exponential decay rates for the fist moment estimate

    beta2: float ∈ [0, 1) default = 0.999
        Exponential decay rates for the second moment estimate

    eps: float = 1e-8
        Term added to the denominator to improve numerical stability

    References
    ----------
    .. [1] Diederik P. Kingma, Jimmy Ba. "Adam: A Method for Stochastic Optimization", 22 Dec 2014
    """
    def __init__(self, model_parameters, weight_decay, lr, beta1, beta2, eps=1e-8):
        super(Adam, self).__init__(
            torch.optim.Adam,
            model_parameters,
            lr=lr,
            betas=[beta1, beta2],
            weight_decay=weight_decay,
            eps=eps,
            amsgrad=False
        )

    @staticmethod
    def get_space():
        return {
            'lr': 'loguniform(1e-5, 1)',
            'beta1': 'loguniform(0.9, 1)',
            'beta2': 'loguniform(0.99, 1)',
            'weight_decay': 'loguniform(1e-10, 1e-3)'
        }

    @staticmethod
    def defaults():
        return {
            'weight_decay': 0.001,
            'lr': 0.001,
            'beta1': 0.9,
            'beta2': 0.999
        }


builders = {'adam': Adam}
