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


class OptimizerAdapter:
    """MixedPrecision Optimizer Adapter
    This handles fp32 & fp16 optimization by providing a common API to both
    """
    def __init__(self, optimizer, half=False, loss_scale=1,
                 dynamic_loss_scale=False, scale_window=1000, scale_factor=2,
                 min_loss_scale=None, max_loss_scale=2.**24):
        if half:
            import apex.fp16_utils.fp16_optimizer as apex_optimizer

            static_loss_scale = loss_scale
            if dynamic_loss_scale:
                static_loss_scale = 'dynamic'

            self.optimizer = apex_optimizer.FP16_Optimizer(
                optimizer,
                static_loss_scale,
                dynamic_loss_scale,
                dynamic_loss_args=dict(
                    init_scale=loss_scale,
                    scale_factor=scale_factor,
                    scale_window=scale_window,
                    min_loss_scale=min_loss_scale,
                    max_loss_scale=max_loss_scale
                ),
                verbose=False
            )
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
        self.half = half

        if half:
            self.model = network_to_half(model)
            self.transform = lambda x: x.half()

    def forward(self, input):
        return self.model(self.transform(input))

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        destination = {
            'model': self.model.state_dict(None, prefix, keep_vars),
            'half': self.half
        }
        return destination

    def load_state_dict(self, state_dict, strict=True):
        self.half = state_dict['half']
        if self.half:
            self.transform = lambda x: x.half()
        self.model.load_state_dict(state_dict['model'], strict=strict)
