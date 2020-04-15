from olympus.optimizers.base import OptimizerInterface


class Noptimizer(OptimizerInterface):
    """No-optimizer Optimizer for when you do not need an optimizer"""
    def __init__(self, *args, call_back=None, **kwargs):
        self.call_back = call_back

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {}

    def load_state_dict(self, state_dict, strict=True):
        return self

    def zero_grad(self):
        return

    def step(self, closure=None):
        if self.call_back:
            self.call_back()
        return

    def backward(self, loss):
        pass

    def add_param_group(self, param_group):
        return

    @staticmethod
    def get_space():
        """Specifies the hyper parameters that are supported by this optimizer"""
        return {}

    @staticmethod
    def defaults():
        """Specifies the hyper parameters defaults"""
        return {}


builders = {
    'no-optimizer': Noptimizer
}
