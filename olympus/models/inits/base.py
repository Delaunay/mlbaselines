import torch
import torch.nn


class Initialization:
    def __call__(self, model):
        """Init model using given function for Linear and Conv2d, and {0, 1} for BatchNorm."""

        for m in model.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                self.layer_init(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                if m.affine:
                    torch.nn.init.constant_(m.weight, 1.0)
                    torch.nn.init.constant_(m.bias, 0.0)

        return model

    def layer_init(self, weight):
        pass


class Uniform(Initialization):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        # self.gen = torch.Generator()
        # self.gen.manual_seed(seed)

    def layer_init(self, weight):
        # jit does not support context managers!
        # if we start using this it will break JIT
        # but we already fork the PRNG with a context manager so probably already breaking jit anyway?
        # with torch.no_grad():
        #   weight.uniform_(self.a, self.b, generator=self.gen)
        torch.nn.init.uniform_(weight, self.a, self.b)

    @staticmethod
    def get_space():
        return {
            'a': 'uniform(0, 1)',
            'b': 'uniform(1, 2)'
        }

    @staticmethod
    def defaults():
        return {
            'a': 0,
            'b': 1
        }


class Normal(Initialization):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def layer_init(self, weight):
        torch.nn.init.normal_(weight, self.mean, self.std)

    @staticmethod
    def get_space():
        return {
            'mean': 'normal(0, 1)',
            'std': 'normal(1, 1)'
        }

    @staticmethod
    def defaults():
        return {
            'mean': 0,
            'std': 1
        }


class Orthogonal(Initialization):
    def __init__(self, gain=1):
        self.gain = gain

    def layer_init(self, weight):
        torch.nn.init.orthogonal_(weight, self.gain)

    @staticmethod
    def get_space():
        return {'gain': 'uniform(0, 1)'}

    @staticmethod
    def defaults():
        return {'gain': 1}


builders = {
    'uniform': Uniform,
    'normal': Normal,
    'orthogonal': Orthogonal}
