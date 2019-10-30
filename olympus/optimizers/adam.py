import torch.optim

from olympus.optimizers.base import OptimizerBuilder


class Adam(OptimizerBuilder):
    def __call__(self, model_parameters, weight_decay, lr, beta1, beta2):
        return torch.optim.Adam(
            model_parameters, lr=lr, betas=[beta1, beta2], weight_decay=weight_decay, eps=1e-8, amsgrad=False)
 
    def get_space(self):
        return {'lr': 'loguniform(1e-5, 1)',
                'beta1': 'loguniform(0.9, 1)', 
                'beta2': 'loguniform(0.99, 1)'}


builders = {'adam': Adam}
