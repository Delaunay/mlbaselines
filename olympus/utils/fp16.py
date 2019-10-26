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
