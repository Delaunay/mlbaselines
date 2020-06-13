import torch
import torch.nn as nn

from olympus.models.module import Module
from olympus.models.inits import Initializer, known_initialization, get_initializers_space

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


def try_convert(x, device, dtype):
    if hasattr(x, 'to'):
        return x.to(device=device, dtype=dtype)

    return x


default_init = Initializer('glorot_uniform', seed=0, gain=1.0)


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
    ...     def __init__(self, input_size, output_size):
    ...         self.main = nn.Linear(input_size[0], output_size[0])
    ...
    ...     def forward(self, x):
    ...         return self.main(x)
    >>>
    >>> model = Model(model=MyModel, input_size=(1, 28, 28), output_size=(10,))

    Raises
    ------
    RegisteredModelNotFound
        when using a name of an known model

    MissingArgument:
        if name nor model were not set
    """
    _dtype = torch.float32
    _device = torch.device('cpu')

    def __init__(self, name=None, *, half=False, model=None, input_size=None, output_size=None,
                 weight_init=default_init, **kwargs):
        super(Model, self).__init__()
        # Save all the args that ware passed down so we can instantiate it again in standalone
        self.replay_args = dict(
            name=name, half=half, model=model, input_size=input_size, output_size=output_size,
            weight_init=weight_init, kwargs=kwargs)

        self.transform = lambda x: try_convert(x, self.device, self.dtype)
        self.half = half
        self._model = None

        # Track defined hyper parameters
        self.hyper_parameters = HyperParameters(space=dict())

        # If init is set then we can add its hyper parameters
        self.weight_init = weight_init
        if weight_init is not None:
            if isinstance(weight_init, str):
                self.weight_init = Initializer(weight_init)

            # replace weight init by its own hyper parameters
            space = self.weight_init.get_space()
            if space:
                self.hyper_parameters.space.update(dict(initializer=space))

        # Make a Lazy Model that will be initialized once all the hyper parameters are set
        if model:
            if hasattr(model, 'get_space'):
                self.hyper_parameters.space.update(model.get_space())

            if isinstance(model, type):
                self.model_builder = LazyCall(
                    model, input_size=input_size, output_size=output_size)
            else:
                self.model_builder = LazyCall(lambda *args, **kwargs: model)

        elif name:
            # load an olympus model
            model_fun = registered_models.get(name)

            if not model_fun:
                raise RegisteredModelNotFound(name)

            self.model_builder = LazyCall(
                model_fun, input_size=input_size, output_size=output_size)

            if hasattr(model_fun, 'get_space'):
                self.hyper_parameters.space.update(model_fun.get_space())
        else:
            raise MissingArgument('Model or Name need to be set')

        # Any Additional parameters set Hyper parameters
        self.other_params = self.hyper_parameters.add_parameters(strict=False, **kwargs)

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def get_space(self):
        """Return hyper parameter space"""
        return self.hyper_parameters.missing_parameters()

    def get_current_space(self):
        """Get currently defined parameter space"""
        return self.hyper_parameters.parameters(strict=False)

    def init(self, override=False, **model_hyperparams):
        others = self.hyper_parameters.add_parameters(strict=False, **model_hyperparams)
        self.other_params.update(others)

        params = self.hyper_parameters.parameters(strict=True)

        initializer = params.pop('initializer', {})
        if isinstance(initializer, dict):
            self.weight_init.init(**initializer)

        self._model = self.model_builder.invoke(**self.other_params, **params)
        self.weight_init(self._model)

        if self.half:
            self._model = network_to_half(self._model)

        # Register module so we can use all the parent methods
        self.add_module('_model', self._model)
        self.replay_args['kwargs'].update(model_hyperparams)
        return self

    @property
    def model(self):
        if not self._model:
            self.init()

        return self._model

    def forward(self, *input, **kwargs):
        return self.model(self.transform(input[0]), *input[1:], **kwargs)

    def __call__(self, *args, **kwargs):
        return super(Model, self).__call__(*args, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        destination = {
            'model': self.model.state_dict(None, prefix, keep_vars),
            'half': self.half,
            'replay': self.replay_args,
            'types': {
                'model': type(self.model)
            }
        }
        return destination

    @staticmethod
    def from_state(state):
        kwargs = state.get('replay')
        kwargs.update(kwargs.pop('kwargs', dict()))
        m = Model(**kwargs)
        m.init()
        m.load_state_dict(state)
        return m

    def load_state_dict(self, state_dict, strict=True):
        self.half = state_dict['half']
        self.model.load_state_dict(state_dict['model'], strict=strict)

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse)

    def to(self, *args, **kwargs):
        self._device, self._dtype, *_ = torch._C._nn._parse_to(*args, **kwargs)
        super(Model, self).to(*args, **kwargs)
        return self

    def act(self, *args, **kwargs):
        return self.model.act(*args, **kwargs)

    def critic(self, *args, **kwargs):
        return self.model.critic(*args, **kwargs)
