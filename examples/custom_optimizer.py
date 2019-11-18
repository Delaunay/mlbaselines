import torch
from torch.optim.optimizer import Optimizer as OptimizerInterface

from olympus.optimizers import Optimizer
from olympus.models import Model


class MySGD(OptimizerInterface):
    def __init__(self, params, lr=0, momentum=0, dampening=0, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay)

        super(MySGD, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay

    @staticmethod
    def get_space():
        return {
            'lr': 'loguniform(1e-5, 1)',
            'momentum': 'uniform(0, 1)',
            'weight_decay': 'loguniform(1e-10, 1e-3)'
        }

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if self.weight_decay != 0:
                    d_p.add_(self.weight_decay, p.data)
                if self.momentum != 0:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(self.momentum).add_(1 - self.dampening, d_p)

                    d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss


if __name__ == '__main__':
    model = Model(
        'logreg',
        input_size=(290,),
        output_size=(10,)
    )

    input = torch.randn((10, 290))

    optimizer = Optimizer(optimizer=MySGD, params=model.parameters())

    # If you use an hyper parameter optimizer, it will generate this for you
    optimizer.init(lr=1e-4, momentum=0.02, weight_decay=1e-3)

    optimizer.zero_grad()

    output = model(input)
    loss = output.sum()
    loss.backward()

    optimizer.step()





