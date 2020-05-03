import torch
import torch.nn as nn

try:
    from olympus.models.bert import BertWrapper
except ImportError:
    class BertWrapper:
        pass


class HalveInputs(nn.Module):

    def __init__(self, inner_model):
        super(HalveInputs, self).__init__()
        self.inner_model = inner_model

    def forward(self, *args, **kwargs):
        args = [x.half() for x in args]
        return self.inner_model(*args)


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
    if isinstance(network, BertWrapper):
        # do not halve the input for BERT - otherwise the embeddings will crash.
        return batchnorm_convert_float(network.half())
    else:
        return HalveInputs(batchnorm_convert_float(network.half()))


def arguments(parser):
    parser.add_argument(
        '--loss-scale', type=float, default=1.0, metavar='SL',
        help='Constant to use to scale loss')
    parser.add_argument(
        '--scale-dynamic', action='store_true', default=True,
        help='Enable dynamic loss scaling')
    parser.add_argument(
        '--scale-factor', type=float, default=2, metavar='SF',
        help='Factor to use to divide or multiply the scale constant')
    parser.add_argument(
        '--scale-window', type=int, default=1000, metavar='SW',
        help='Number of batches to wait before increasing the scale factor')
    parser.add_argument(
        '--scale-min', type=float, default=1, metavar='SMN',
        help='Minimum scaling factor')
    parser.add_argument(
        '--scale-max', type=float, default=2.**24, metavar='SMX',
        help='Maximum scaling factor')
    return parser
