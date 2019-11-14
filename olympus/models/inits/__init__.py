from olympus.utils.factory import fetch_factories
from olympus.utils import seed as init_seed, warning

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
