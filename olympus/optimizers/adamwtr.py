from transformers.optimization import AdamW as HugginsAdamW

from olympus.optimizers.base import OptimizerAdapter


class AdamW(OptimizerAdapter):
    """Implements Adam algorithm with weight decay fix
   
    Just adding the square of the weights to the loss function is *not*
    the correct way of using L2 regularization/weight decay with Adam but it is what
    implementation do. This is the fixed version.
    
    From `transformer <https://huggingface.co/transformers/_modules/transformers/optimization.html#AdamW>`_.
    
    See also :class`.Adam`
    """
    def __init__(self, model_parameters, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-6,
                 weight_decay=0.0, correct_bias=True):
        super(AdamW, self).__init__(
            HugginsAdamW,
            model_parameters,
            lr=lr,
            betas=[beta1, beta2],
            weight_decay=weight_decay,
            eps=eps,
        )

    @staticmethod
    def get_space():
        return {
            'lr': 'loguniform(1e-5, 1)',
            'beta1': 'loguniform(0.8, 1)',
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


builders = {'adamwtr': AdamW}
