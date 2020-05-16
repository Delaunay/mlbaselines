import random
import time
from typing import Callable, Optional, TypeVar, Dict, NoReturn, Union
from urllib.parse import urlparse

import base64
import bson
import zlib

import numpy

import torch

from olympus.utils.options import option, set_option
from olympus.utils.chrono import Chrono
from olympus.utils.functional import select, flatten
from olympus.utils.arguments import parse_args, show_hyperparameter_space, required, get_parameters, drop_empty_key
from olympus.utils.log import warning, info, debug, error, critical, exception, set_verbose_level, set_log_level

A = TypeVar('A')
R = TypeVar('R')


class MissingArgument(Exception):
    pass


def fetch_device():
    """Set the default device to CPU if cuda is not available"""
    default = 'cpu'
    if torch.cuda.is_available():
        default = 'cuda'

    return torch.device(option('device.type', default))


def show_dict(dictionary: Dict, indent: int = 0) -> NoReturn:
    print(' ' * indent + '-' * 80)
    for k, v in dictionary.items():
        print(f'{k:>30}: {v}')
    print(' ' * indent + '-' * 80)


def compress_dict(state: Dict) -> Dict:
    """Compress a state dictionary and return a json friendly compressed state"""
    binary = bson.encode(state)
    compressed_json = base64.b64encode(zlib.compress(binary))
    crc32 = zlib.crc32(binary)
    return dict(zlib=compressed_json, crc32=crc32)


def decompress_dict(state: Dict) -> Dict:
    """Decompress a state dictionary and return its json"""
    if 'zlib' in state:
        binary = base64.b64decode(state['zlib'])
        decompressed_bson = zlib.decompress(binary)
        assert zlib.crc32(decompressed_bson) == state['crc32'], 'State is corrupted'

        return bson.decode(decompressed_bson)

    return state


class TimeThrottler:
    """Limit how often the function `fun` is called in seconds

    Examples
    --------

    .. code-block:: python

        throttled_print = TimeThrottler(print, every=1)

        # Only prints 0
        for i in range(0, 10):
            throttled_print(i)

        # Prints 0 to 9
        for i in range(0, 10):
            throttled_print(i)
            time.sleep(1)
    """
    def __init__(self, fun: Callable[[A], R], every=10):
        self.fun = fun
        self.last_time: float = 0
        self.every: float = every

    def __call__(self, *args, **kwargs) -> Optional[R]:
        now = time.time()
        elapsed = now - self.last_time

        if elapsed > self.every:
            self.last_time = now
            return self.fun(*args, **kwargs)

        return None


def parse_uri_options(options: str) -> Dict:
    if not options:
        return dict()

    opt = dict()

    for item in options.split('&'):
        k, v = item.split('=')
        opt[k] = v

    return opt


def parse_uri(uri: str) -> Dict:
    parsed = urlparse(uri)
    netloc = parsed.netloc

    arguments = {
        'scheme': parsed.scheme,
        'path': parsed.path,
        'query': parse_uri_options(parsed.query),
        'fragment': parsed.fragment,
        'params': parsed.params
    }

    if netloc:
        usr_pwd_add_port = netloc.split('@')

        if len(usr_pwd_add_port) == 2:
            usr_pwd = usr_pwd_add_port[0].split(':')
            if len(usr_pwd) == 2:
                arguments['password'] = usr_pwd[1]
            arguments['username'] = usr_pwd[0]

        add_port = usr_pwd_add_port[-1].split(':')
        if len(add_port) == 2:
            arguments['port'] = add_port[1]
        arguments['address'] = add_port[0]

    return arguments


def get_value(item: Union[float, torch.Tensor]) -> float:
    if isinstance(item, torch.Tensor):
        return item.item()
    return item


def find_batch_size(model, shape, low, high, dtype=torch.float32):
    """Find the highest batch size that can fit in memory using binary search"""
    low = (low // 8) * 8
    high = (1 + high // 8) * 8

    batches = list(range(low, high, 8))

    a = 0
    b = len(batches)
    mid = a + (b - a) // 2

    while b != a + 1:
        mid = a + (b - a) // 2

        try:
            batch_size = batches[mid]
            tensor = torch.randn((batch_size,) + shape, dtype=dtype)

            model(tensor)

            # ran successfully
            a = mid

        # ran out of memory
        except RuntimeError as e:
            if 'out of memory' in str(e):
                b = mid
            else:
                raise e

    return batches[mid]


class CircularDependencies(Exception):
    pass


class LazyCall:
    """Save the call parameters of a function for it can be invoked at a later date"""
    def __init__(self, fun, *args, **kwargs):
        self.fun = fun
        self.args = args
        self.kwargs = kwargs
        self.obj = None
        self.is_processing = False

    def __call__(self, *args, **kwargs):
        self.invoke()
        return self.obj(*args, **kwargs)

    def add_arguments(self, *args, **kwargs):
        self.args = self.args + args
        self.kwargs.update(kwargs)

    def invoke(self, **kwargs):
        if self.obj is None:
            self.is_processing = True
            self.obj = self.fun(*self.args, **self.kwargs, **kwargs)
            self.is_processing = False
            return self.obj
        return self.obj

    def __getattr__(self, item):
        if self.obj is None and self.is_processing:
            raise CircularDependencies('Circular dependencies')

        self.invoke()
        return getattr(self.obj, item)

    def was_invoked(self):
        return self.obj is not None


class MissingParameters(Exception):
    pass


class WrongParameter(Exception):
    pass


def missing_params(space, kwargs, missing):
    for k, v in space.items():

        if not isinstance(v, dict):
            if k not in kwargs:
                missing[k] = v

        else:
            if k not in kwargs:
                missing[k] = v
                continue

            if k not in missing:
                missing[k] = {}

            sub_missing = missing_params(v, kwargs[k], missing[k])

            # If nothing is missing pop it
            if len(sub_missing) == 0:
                missing.pop(k)

    return missing


def update_params(space, kwargs, params):
    for k, v in kwargs.items():
        if k not in space:
            raise WrongParameter(f'{k} is not a valid parameter, pick from: {space.keys()}')

        if isinstance(v, dict):
            if k not in params:
                params[k] = {}

            update_params(space[k], v, params[k])
        else:
            params[k] = v


class HyperParameters:
    """Keeps track of mandatory hyper parameters

    Parameters
    ----------
    space: Dict[str, Space]
        A dictionary defining each parameters and their respective space/dim

    kwargs:
        A dictionary of defined hyper parameters
    """
    def __init__(self, space, **kwargs):
        self.space = space
        self.current_parameters = {}
        self.add_parameters(**kwargs)

    def missing_parameters(self):
        """Returns a dictionary of missing parameters"""
        return missing_params(self.space, self.current_parameters, {})

    def add_parameters(self, **kwargs):
        """Insert a new parameter value"""
        update_params(self.space, kwargs, self.current_parameters)

    def parameters(self, strict=False):
        """Returns all the parameters and checks if any are missing"""
        if strict:
            missing = self.missing_parameters()
            if missing:
                raise MissingParameters('Parameters are missing: {}'.format(', '.join(missing.keys())))

        return self.current_parameters


# Tracks global seeds
SEEDS = {}


def new_seed(**kwargs):
    """Global seed management"""
    global SEEDS
    import random
    assert len(kwargs) == 1, 'Only single seed can be registered'

    # Allow user to force seed to change seeds automatically each time the program is ran
    # Disabled by default
    automatic_seeding = option('seeding.random', default=False, type=bool)

    for name, value in kwargs.items():
        # do not change the seed if it was already set
        if name in SEEDS:
            return SEEDS[name]

        elif not automatic_seeding:
            SEEDS[name] = value

        else:
            val = random.getrandbits(64)
            SEEDS[name] = val
            kwargs[name] = val

    return kwargs.popitem()[1]


def get_seeds():
    """Returns a set of seed that are used by the program"""
    return SEEDS


def set_seeds(seed):
    """Set most commonly used global seeds"""
    print('seed:', seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def get_rng_states():
    state = dict()
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()

    state['random'] = random.getstate()
    state['numpy'] = numpy.random.get_state()
    state['torch_cpu'] = torch.get_rng_state()

    return state


def set_rng_states(state):
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    elif 'torch_cuda' in state:
        raise RuntimeError('Cannot restore state without a GPU.')

    random.setstate(state['random'])
    numpy.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
