from olympus.utils.factory import fetch_factories

from olympus.utils import set_seeds as init_seed, warning, HyperParameters

from torch.nn import Module
from torch.random import fork_rng

registered_initialization = fetch_factories('olympus.models.inits', __file__)


def known_initialization():
    return registered_initialization.keys()


def register_initialization(name, factory, override=False):
    global registered_initialization

    if name in registered_initialization:
        warning(f'{name} was already registered, use override=True to ignore')

        if not override:
            return

    registered_initialization[name] = factory


class RegisteredInitNotFound(Exception):
    pass


def get_initializers_space():
    space = {}
    for k, initializer in registered_initialization.items():
        space[k] = initializer.get_space()

    return dict(initializer=space)


class Initializer:
    """Lazy Initializer"""
    def __init__(self, name, seed=0, **kwargs):
        self.name = name
        self.hyper_parameters = HyperParameters(space={})
        self.seed = seed
        self._initializer = None

        self.initializer_ctor = registered_initialization.get(name)

        if self.initializer_ctor is None:
            raise RegisteredInitNotFound(name)

        if hasattr(self.initializer_ctor, 'get_space'):
            self.hyper_parameters.space = self.initializer_ctor.get_space()

        self.hyper_parameters.add_parameters(**kwargs)

    def get_space(self):
        """Return the dimension space of each parameters"""
        return self.hyper_parameters.missing_parameters()

    def get_current_space(self):
        """Get currently defined parameter space"""
        return self.hyper_parameters.parameters(strict=False)

    def init(self, override=False, **kwargs):
        if self._initializer and not override:
            warning('Initializer is already set, use override=True to force re initialization')
            return self

        self.hyper_parameters.add_parameters(**kwargs)
        self._initializer = self.initializer_ctor(**self.hyper_parameters.parameters(strict=True))

        return self

    @property
    def initializer(self):
        if not self._initializer:
            self.init()

        return self._initializer

    def __call__(self, model):
        with fork_rng(enabled=True):
            init_seed(self.seed)

            return self.initializer(model)


def initialize_weights(model, name=None, seed=0, half=False, **kwargs):
    """Initialize the model weights using a particular method

    Parameters
    ----------
    model: Module

    name: str
        Name of the initializer

    seed: int
        seed to use for the PRNGs

    Returns
    -------
    The initialized model
    """
    # TODO: remove dependency to global PRNG
    # At the moment we simply fork the PRNG to prevent affecting later calls
    with fork_rng(enabled=True):
        init_seed(seed)

        method_builder = registered_initialization.get(name)

        if not method_builder:
            raise RegisteredInitNotFound(name)

        method = method_builder(**kwargs)
        return method(model)
