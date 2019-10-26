import torch.nn as nn
import torch.optim as optim


class OptimizerAdapter:
    def __init__(self, optimizer: optim.Optimizer, **kwargs):
        self.optimizer = optimizer

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self, cost=None, closure=None, **kwargs):
        return self.optimizer.step(closure)

    def backward(self, cost: nn.Module):
        return cost.backward()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, img):
        return self.optimizer.load_state_dict(img)


class Fp16OptimizerAdapter:
    def __init__(self, optimizer: optim.Optimizer, **kwargs):
        from apex import amp

        self.optimizer = optimizer
        self.amp = amp.init()

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self, cost=None, closure=None, **kwargs):
        return self.optimizer.step(closure)

    def backward(self, cost: nn.Module):
        with self.amp.scale_loss(cost, self.optimizer) as loss:
            return loss.backward()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, img):
        return self.optimizer.load_state_dict(img)

