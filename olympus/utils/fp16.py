import torch
import torch.nn as nn


class ImplicitFp16Cast(nn.Module):
    def __init__(self):
        super(ImplicitFp16Cast, self).__init__()

    def forward(self, input):
        return input.half()


def copy_in_params(net, params):
    net_params = list(net.parameters())

    for i in range(len(params)):
        net_params[i].data.copy_(params[i].data)


def set_grad(params, params_with_grad):
    for param, param_w_grad in zip(params, params_with_grad):

        if param.grad is None:
            param.grad = torch.nn.Parameter(param.data.new().resize_(*param.data.size()))

        param.grad.data.copy_(param_w_grad.grad.data)


def get_param_copy(net):
    param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in net.parameters()]

    for param in param_copy:
        param.requires_grad = True

    return param_copy


def batchnorm_convert_float(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()

    for child in module.children():
        batchnorm_convert_float(child)

    return module


def network_to_half(network):
    return nn.Sequential(ImplicitFp16Cast(), batchnorm_convert_float(network.half()))


class OptimizerAdapter:
    """MixedPrecision Optimizer Adapter
    This handles fp32 & fp16 optimization by providing a common API to both
    """
    def __init__(self, optimizer, half=False, *args, **kwargs):
        if half:
            import apex.fp16_utils.fp16_optimizer as apex_optimizer
            self.optimizer = apex_optimizer.FP16_Optimizer(optimizer, *args, **kwargs)
        else:
            self.optimizer = optimizer

        self.half = half

    def backward(self, loss):
        if self.half:
            self.optimizer.backward(loss)
        else:
            loss.backward()

        return self.optimizer

    def step(self):
        return self.optimizer.step()

    def zero_grad(self):
        return self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def __getattr__(self, item):
        return getattr(self.optimizer, item)


class ModelAdapter(nn.Module):
    """Add a fp16 cast at the beginning of the model to make switching between fp16 & fp32
    seamless
    """
    def __init__(self, model, half=False):
        super(ModelAdapter, self).__init__()

        self.model = model
        self.transform = lambda x: x

        if half:
            self.model = network_to_half(model)
            self.transform = lambda x: x.half()

    def forward(self, input):
        return self.model(self.transform(input))

    def __getattr__(self, item):
        return getattr(self.model, item)
