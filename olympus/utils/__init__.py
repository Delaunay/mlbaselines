import random
import time
from typing import Callable, Optional, TypeVar
from urllib.parse import urlparse

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


def show_dict(dictionary, indent=0):
    print(' ' * indent + '-' * 80)
    for k, v in dictionary.items():
        print(f'{k:>30}: {v}')
    print(' ' * indent + '-' * 80)


class TimeThrottler:
    """Limit how often the function `fun` is called in seconds"""
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


def parse_uri_options(options):
    if not options:
        return dict()

    opt = dict()

    for item in options.split('&'):
        k, v = item.split('=')
        opt[k] = v

    return opt


def parse_uri(uri):
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


def get_storage(uri, objective=None):
    """Shorten the storage config from orion that is super long an super confusing
        <storage_type>:<database>:<file or address>

        legacy:pickleddb:my_data.pkl
        legacy:mongodb://user@pass:192.168.0.0:8989
    """
    storage_type, storage_uri = uri.split(':', maxsplit=1)
    arguments = parse_uri(storage_uri)
    database = arguments.get('scheme', 'pickleddb')
    database_resource = arguments.get('path', arguments.get('address'))

    if storage_type == 'legacy':
        # TODO: make it work for mongodb
        return {
            'type': storage_type,
            'database': {
                'type': database,
                'host': database_resource,
            }
        }

    if storage_type == 'track':
        return {
            'type': 'track',
            'uri': f'{storage_uri}?objective={objective}'
        }


def get_value(item):
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
        self.check_correct_parameters(kwargs)
        self.current_parameters = kwargs

    def check_correct_parameters(self, kwargs):
        for k, v in kwargs.items():
            if k not in self.space:
                raise WrongParameter(f'{k} is not a valid parameter, pick from: {self.space.keys()}')

    def missing_parameters(self):
        missing = {}
        for k, v in self.space.items():
            if k not in self.current_parameters:
                missing[k] = v

        return missing

    def add_parameters(self, **kwargs):
        self.check_correct_parameters(kwargs)
        self.current_parameters.update(kwargs)

    def parameters(self, strict=False):
        if strict:
            missing = self.missing_parameters()
            if missing:
                raise MissingParameters('Parameters are missing: {}'.format(', '.join(missing.keys())))

        return self.current_parameters


SEEDS = {}


def new_seed(**kwargs):
    """Global seed management"""
    global SEEDS
    import random
    assert len(kwargs) == 1, 'Only single seed can be registered'

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
    return SEEDS


def set_seeds(seed):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(new_seed(torch_cuda=seed))
        # torch.backends.cudnn.deterministic = True

    random.seed(new_seed(python_rand=seed))
    numpy.random.seed(new_seed(numpy=seed))
    torch.manual_seed(new_seed(torch_cpu=seed))
