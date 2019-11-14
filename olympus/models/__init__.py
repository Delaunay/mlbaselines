import torch
import torch.nn as nn

from olympus.models.inits import initialize_weights, known_initialization
from olympus.utils import MissingArgument, warning, LazyCall, HyperParameters
from olympus.utils.factory import fetch_factories
from olympus.utils.fp16 import network_to_half

registered_models = fetch_factories('olympus.models', __file__)


def known_models():
    return registered_models.keys()


def register_model(name, factory, override=False):
    global registered_models

    if name in registered_models:
        warning(f'{name} was already registered, use override=True to ignore')

        if not override:
            return

    registered_models[name] = factory


class RegisteredModelNotFound(Exception):
    pass


class Module(nn.Module):
    """Olympus Module interface to guide new users when doing NAS"""
    def __init__(self, input_size=None, output_size=None):
        super(Module, self).__init__()

    @staticmethod
    def get_space():
        raise NotImplemented


def try_convert(x, device, dtype):
    if hasattr(x, 'to'):
        return x.to(device=device, dtype=dtype)

    return x


# TODO: Make Model distributed here
class Model(nn.Module):
    """Olympus standardized Model interface

    Parameters
    ----------
    name: str
        Name of a registered model

    half: bool
        Convert the network to half/fp16

    model: Model
        Custom model to use, mutually exclusive with :param name

    Examples
    --------

    Model wrappers that provide a wide range of utility built-in.

    Can instantiate common model directly

    >>> model = Model('resnet18', input_size=(1, 28, 28), output_size=(10,))

    Handles mixed precision conversion for you

    >>> model = Model('resnet18', input_size=(1, 28, 28), output_size=(10,), half=True)

    Handles weight initialization

    >>> model = Model('resnet18', input_size=(1, 28, 28), output_size=(10,), weight_init='glorot_uniform')

    Supports your custom model

    >>> class MyModel(nn.Module):
    >>>     def __init__(self, input_size, output_size):
    >>>         self.main = nn.Linear(input_size[0], output_size[0])
    >>>
    >>>     def forward(self, x):
    >>>         return self.main(x)
    >>>
    >>> model = Model(MyModel, input_size=(1, 28, 28), output_size=(10,))

    Raises
    ------
    RegisteredModelNotFound
        when using a name of an known model

    MissingArgument:
        if name nor model were not set
    """
    MODEL_BASE_SPACE = {
        'weight_init': 'choices({})'.format(list(known_initialization()))
    }
    _dtype = torch.float32
    _device = torch.device('cpu')

    def __init__(self, name=None, *, half=False, model=None, input_size=None, output_size=None, weight_init=None, seed=0):
        super(Model, self).__init__()
        self.transform = lambda x: try_convert(x, self.device, self.dtype)
        self.half = half
        self.seed = seed
        self._model = None

        # Track defined hyper parameters
        self.hyper_parameters = HyperParameters(space=Model.MODEL_BASE_SPACE)
        if weight_init:
            self.hyper_parameters.add_parameters(weight_init=weight_init)

        # Make a Lazy Model that will be initialized once all the hyper parameters are set
        if model:
            if hasattr(model, 'get_space'):
                self.hyper_parameters.space.update(model.get_space())

            self.model_builder = LazyCall(lambda x: model)

        elif name:
            # load an olympus model
            model_fun = registered_models.get(name)

            if not model_fun:
                raise RegisteredModelNotFound(name)

            self.model_builder = LazyCall(model_fun, input_size=input_size, output_size=output_size)

            if hasattr(model_fun, 'get_space'):
                self.hyper_parameters.space.update(model.get_space())
        else:
            raise MissingArgument('Model or Name needs to be set')

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def get_space(self):
        """Return hyper parameter space"""
        return self.hyper_parameters.missing_parameters()

    def init(self, override=False, weight_init=None):
        self.model_builder.invoke()
        self._model = self.model_builder.obj

        parameters = self.hyper_parameters.parameters(strict=False)
        if weight_init is not None:
            parameters['weight_init'] = weight_init

        init_name = parameters['weight_init']
        initialize_weights(self._model, name=init_name, seed=self.seed)

        if self.half:
            self._model = network_to_half(self._model)

        # Register module so we can use all the parent methods
        self.add_module('main_model', self._model)

        return self

    @property
    def model(self):
        if not self._model:
            self.init()

        return self._model

    def forward(self, *input, **kwargs):
        return self.model(self.transform(input[0]), *input[1:], **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        destination = {
            'model': self.model.state_dict(None, prefix, keep_vars),
            'half': self.half
        }
        return destination

    def load_state_dict(self, state_dict, strict=True):
        self.half = state_dict['half']
        self.model.load_state_dict(state_dict['model'], strict=strict)

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse)

    def to(self, *args, **kwargs):
        self._device, self._dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        super().to(*args, **kwargs)
        return self
