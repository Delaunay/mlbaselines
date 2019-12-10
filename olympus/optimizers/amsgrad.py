import torch.optim

from olympus.optimizers.base import OptimizerAdapter


class AMSGrad(OptimizerAdapter):
    """Variant of Adam

    See also :class`.Adam`

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
    .. [1] Tran Thi Phuong, Le Trieu Phong. "On the Convergence Proof of AMSGrad and a New Version", 7 Apr 2019
    """
    def __init__(self, model_parameters, weight_decay, lr, beta1, beta2, eps=1e-8):
        super(AMSGrad, self).__init__(
            torch.optim.Adam,
            model_parameters,
            lr=lr,
            betas=[beta1, beta2],
            weight_decay=weight_decay,
            eps=eps,
            amsgrad=True
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


builders = {'amsgrad': AMSGrad}
