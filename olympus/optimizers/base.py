import torch.optim as optim


class OptimizerInterface(optim.Optimizer):
    """Base Olympus Optimizer"""

    def __init__(self, params):
        super(OptimizerInterface, self).__init__(params)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return super(OptimizerInterface, self).state_dict()

    def load_state_dict(self, state_dict, strict=True):
        return super(OptimizerInterface, self).load_state_dict(state_dict)

    def zero_grad(self):
        return super(OptimizerInterface, self).zero_grad()

    def step(self, closure=None):
        return super(OptimizerInterface, self).step(closure)

    def backward(self, loss):
        """This method comes from FP16 Optimizer, for consistency we add it everywhere"""
        loss.backward()

    def add_param_group(self, param_group):
        return super(OptimizerInterface, self).add_param_group(param_group)

    @staticmethod
    def get_space():
        """Specifies the hyper parameters that are supported by this optimizer"""
        return {}

    @staticmethod
    def defaults():
        """Specifies the hyper parameters defaults"""
        return {}


class OptimizerAdapter(OptimizerInterface):
    """Wraps an existing Pytorch Optimizer into an Olympus optimizer"""

    def __init__(self, factory, *args, **kwargs):
        self.optimizer = factory(*args, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        return self.optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self, closure=None):
        return self.optimizer.step(closure)

    def backward(self, loss):
        """This method comes from FP16 Optimizer, for consistency we add it everywhere"""
        loss.backward()

    def add_param_group(self, param_group):
        return self.optimizer.add_param_group(param_group)

    @property
    def state(self):
        return self.optimizer.state

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @staticmethod
    def get_space():
        """Specifies the hyper parameters that are supported by this optimizer"""
        raise NotImplementedError()

    @staticmethod
    def defaults():
        """Specifies the hyper parameters defaults"""
        raise NotImplementedError()
