from typing import Dict

import torch
from torch.optim.optimizer import Optimizer as TorchOptimizer

from olympus.utils import MissingArgument, warning, HyperParameters
from olympus.utils.factory import fetch_factories


registered_optimizers = fetch_factories('olympus.optimizers', __file__)


def known_optimizers():
    return registered_optimizers.keys()


class RegisteredOptimizerNotFound(Exception):
    pass


class UninitializedOptimizer(Exception):
    pass


def register_optimizer(name, factory, override=False):
    global registered_optimizers

    if name in registered_optimizers:
        warning(f'{name} was already registered, use override=True to ignore')

        if not override:
            return

    registered_optimizers[name] = factory


class Optimizer(TorchOptimizer):
    """Lazy Optimizer that allows you to first fetch the supported parameters using ``get_space`` and then
    initialize the underlying optimizer using ``init_optimizer``

    Parameters
    ----------
    name: str
        Name of a registered optimizer

    optimizer: Optimizer
        Custom optimizer, mutually exclusive with :param name

    half: bool
        Enable fp16 Optimizer

    loss_scale: float (LS)
        fp16 optimizer option: loss scale to use

    dynamic_loss_scale: bool
        fp16 optimizer option: Enable dynamic loss scaling

    scale_window: int (SW)
        dynamic loss scaling option: Increase LS after SW successful iteration

    scale_factor: float (SF)
        dynamic loss scaling option: divide LS by SF after an overflow, or
        multiply LS by SF after SW successful iteration

    min_loss_scale: float

    max_loss_scale: float

    Examples
    --------

    Follows standard Pytorch Optimizer

    >>> optimizer = Optimizer('SGD', model.parameters(),  weight_decay, lr=0.001, momentum=0.8)
    >>> optimizer.zero_grad()
    >>> loss = model(x)
    >>> optimizer.backward(loss)
    >>> optimizer.step()

    Can be lazily initialized for hyper parameter search

    >>> optimizer = Optimizer('SGD')
    >>> optimizer.get_space()
    {'lr': 'loguniform(1e-5, 1)', 'momentum': 'uniform(0, 1)'}
    >>> optimizer.init_optimizer(model.parameters(), weight_decay, lr=0.001, momentum=0.8)
    >>> optimizer.zero_grad()
    >>> loss = model(x)
    >>> optimizer.backward(loss)
    >>> optimizer.step()

    Switch to a mixed precision optimizer if needed

    >>> optimizer = Optimizer('SGD', half=True)

    Raises
    ------
    RegisteredOptimizerNotFound
        when using a name of an known optimizers

    MissingArgument:
        if name nor optimizer were not set

    WrongParameter
        if a wrong hyper parameter is passed in kwargs
    """
    half = False
    half_args = dict()
    _optimizer = None

    def __init__(self, name=None, *, params=None, optimizer=None, half=False, loss_scale=1,
                 dynamic_loss_scale=False, scale_window=1000, scale_factor=2,
                 min_loss_scale=None, max_loss_scale=2.**24, **kwargs):
        self._optimizer = None
        self._model_parameters = params
        self._half_parameters(
            half,  loss_scale, dynamic_loss_scale,
            scale_window, scale_factor, min_loss_scale, max_loss_scale
        )

        # Track defined hyper parameters
        self.hyper_parameters = HyperParameters(space={})

        if optimizer:
            warning('Using custom optimizer')
            self._optimizer = self._wrap_optimizer(optimizer)

            if hasattr(self._optimizer, 'get_space'):
                self.hyper_parameters.space = self._optimizer.get_space()

        elif name:
            # load an olympus model
            self.optimizer_builder = registered_optimizers.get(name.lower())

            if not self.optimizer_builder:
                raise RegisteredOptimizerNotFound(name)

            self.optimizer_builder = self.optimizer_builder()

            if hasattr(self.optimizer_builder, 'get_space'):
                self.hyper_parameters.space = self.optimizer_builder.get_space()

        else:
            raise MissingArgument('optimizer or name needs to be set')

        # All additional args are hyper parameters
        self.hyper_parameters.add_parameters(**kwargs)

    def _half_parameters(self, half=False, loss_scale=1,
                         dynamic_loss_scale=False, scale_window=1000, scale_factor=2,
                         min_loss_scale=None, max_loss_scale=2.**24):
        """Save the configuration of the fp16 optimizer"""
        self.half = half

        static_loss_scale = loss_scale
        if dynamic_loss_scale:
            static_loss_scale = 'dynamic'

        self.half_args = dict(
            static_loss_scale=static_loss_scale,
            dynamic_loss_scale=dynamic_loss_scale,
            dynamic_loss_args=dict(
                init_scale=loss_scale,
                scale_factor=scale_factor,
                scale_window=scale_window,
                min_loss_scale=min_loss_scale,
                max_loss_scale=max_loss_scale
            ),
            verbose=False
        )

    def _wrap_optimizer(self, optimizer):
        if self.half:
            import apex.fp16_utils.fp16_optimizer as apex_optimizer
            return apex_optimizer.FP16_Optimizer(optimizer, **self.half_args)

        return optimizer

    def get_space(self) -> Dict[str, str]:
        """Return the dimension space of each parameters"""
        if self._optimizer:
            warning('Optimizer is already set')

        return self.hyper_parameters.missing_parameters()

    def get_params(self, params: Dict[str, any]) -> Dict[str, any]:
        """Extract optimizer parameters from the dictionary, the resulting dictionary can be
        used to initialize the optimizer without worry.
        """
        if self._optimizer:
            warning('Optimizer is already set!')

        if self.optimizer_builder:
            return self.optimizer_builder.get_params(params)

        return {}

    def init(self, model_parameters=None, override=False, **kwargs):
        """instantiate the underlying optimizer

        Raises
        ------
        MissingParameters
            if an hyper parameter is missing
        """
        if self._optimizer and not override:
            warning('Optimizer is already set, use override=True to force re initialization')
            return self

        # add missing hyper parameters
        self.hyper_parameters.add_parameters(**kwargs)

        if model_parameters is None:
            model_parameters = self._model_parameters

        if model_parameters is None:
            raise MissingArgument('Missing Model parameters!')

        self._optimizer = self._wrap_optimizer(
            self.optimizer_builder(model_parameters, **self.hyper_parameters.parameters(strict=True)))

        return self

    @property
    def optimizer(self):
        if not self._optimizer:
            self.init()

        return self._optimizer

    def backward(self, loss):
        if self.half:  # for loss scaling
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

    @property
    def defaults(self):
        if self.half:
            return self.optimizer.optimizer.defaults

        return self.optimizer.defaults

    @property
    def state(self):
        return self.optimizer.state

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, device=None):
        self.optimizer.load_state_dict(state_dict)

        # Dirty fix found here:
        # https://github.com/pytorch/pytorch/issues/2830#issuecomment-336194949
        if device:
            for state in self.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device=device)

