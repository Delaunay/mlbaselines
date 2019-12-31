from argparse import Namespace, ArgumentParser
from collections import defaultdict
import logging
import random
import sys
import time
from typing import Callable, Optional, TypeVar
from urllib.parse import urlparse

import numpy

import torch

from olympus.utils.options import option, set_option
from olympus.utils.chrono import Chrono
from olympus.utils.functional import select, flatten

A = TypeVar('A')
R = TypeVar('R')


class MissingArgument(Exception):
    pass


def drop_empty_key(space):
    new_space = {}
    for key, val in space.items():
        if val:
            new_space[key] = val

    return new_space


def get_parameters(name, params):
    if params is None:
        return {}
    return params.get(name, {})


def _insert_hyperparameter(hypers_dict, name, value):
    """Insert an hyper parameter inside the dictionary"""
    name, hp_name = name.replace('--', '').split('.', maxsplit=1)
    data = hypers_dict.get(name, {})

    try:
        data[hp_name] = float(value)
    except:
        data[hp_name] = value

    hypers_dict[name] = data


def parse_arg_file(arg_file, parser, args, hypers_dict):
    """Parse a json file, command line override configuration file

    Examples
    --------
    >>> {
    >>>     'model': 'resnet18',
    >>>     'optimizer': {'sgd':{
    >>>         'lr': 0.001
    >>>     }}
    >>>     'schedule': 'none',
    >>>     'optimizer.momentum': 0.99
    >>> }

    """
    import json
    arguments = json.load(open(arg_file, 'r'))

    for arg_name, arg_value in arguments.items():
        # This is an hyper parameter
        if arg_name.find('.') != -1:
            _insert_hyperparameter(hypers_dict, arg_name, arg_value)

        # Argument with Hyper parameters
        elif isinstance(arg_value, dict):
            assert len(arg_value) == 1, 'Hyper parameter dict should only have one mapping'
            name, parameters = list(arg_value.items())[0]
            args[arg_name] = name

            for param_name, param_value in parameters.items():
                _insert_hyperparameter(hypers_dict, f'{arg_name}.{param_name}', param_value)

        # Simple argument => Value
        else:
            default_val = parser.get_default(arg_name)

            # is the arguments is not set, we use the file override
            val = args.get(arg_name)

            if val is None or val == default_val:
                val = arg_value
            # else we keep the argument value from the command line
            args[arg_name] = val


# we have to create our own required argument system in case the required argument
# is provided inside the configuration file
class required:
    pass


def parse_args(parser: ArgumentParser, script_args=None):
    """Parse known args assume the additional arguments are hyper parameters"""
    args, hypers = parser.parse_known_args(script_args)

    args = vars(args)
    hypers_dict = dict()

    # File Override
    arg_file = args.get('arg_file')
    if arg_file is not None:
        parse_arg_file(arg_file, parser, args, hypers_dict)

    # Hyper Parameters
    i = 0
    try:
        while i < len(hypers):
            _insert_hyperparameter(hypers_dict, hypers[i], hypers[i + 1])
            i += 2
    except Exception as e:
        error(f'Tried to parse hyper parameters but {e} occurred')
        raise e

    args['hyper_parameters'] = hypers_dict
    for k, v in args.items():
        if isinstance(v, required):
            raise RuntimeError(f'Argument {k} is required!')

    return Namespace(**args)


def display_space(space, ss):
    for type, methods in space.items():
        print(f'  {type.capitalize()}', file=ss)
        print(f'  {"~" * len(type)}', file=ss)

        for method, hps in methods.items():
            print(f'    {method}', file=ss)

            for hyper_name, space in hps.items():
                print(f'      --{type}.{hyper_name:<20}: {space}', file=ss)

    print(file=ss)


def show_hyperparameter_space():
    import io
    from olympus.models import get_initializers_space
    from olympus.optimizers import get_schedules_space, get_optimizers_space

    ss = io.StringIO()
    print('conditional arguments:', file=ss)

    display_space(get_initializers_space(), ss)
    display_space(get_optimizers_space(), ss)
    display_space(get_schedules_space(), ss)
    txt = ss.getvalue()
    ss.close()
    return txt


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


verbose_log_mapping = [
    logging.WARN,   # 0
    logging.WARN,   # 1
    logging.INFO,   # 2
    logging.DEBUG   # 3
]


def set_verbose_level(level):
    """Set verbose level
        - 0 disables all progress output (warning logging enabled)
        - 1 enables progress output and warning logging
        - 2 adds info logging
        - 3 adds debug logging
    """
    if level <= 0:
        # mute progress printing
        set_option('progress.frequency_epoch', 0)
        set_option('progress.frequency_batch', 0)

    if level >= len(verbose_log_mapping):
        level = -1

    set_log_level(verbose_log_mapping[level])


def set_log_level(level=logging.INFO):
    oly_log.setLevel(level)


def log_record(name, level, path, lno, msg, args, exc_info, func=None, sinfo=None, **kwargs):
    start = path.rfind('/olympus/')
    if start > -1:
        path = path[start:]
    return logging.LogRecord(name, level, path, lno, msg, args, exc_info, func, sinfo, **kwargs)


def make_logger(name):
    logger = logging.getLogger(name)
    logger.propagate = False
    logging.setLogRecordFactory(log_record)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.stream = sys.stdout

    formatter = logging.Formatter(
        '%(relativeCreated)8d [%(levelname)8s] %(name)s [%(process)d] %(pathname)s:%(lineno)d %(message)s')
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(ch)

    return logger


def parse_options(options):
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
        'query': parse_options(parsed.query),
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


if globals().get('oly_log') is None:
    oly_log = make_logger('OLY')
    set_log_level(option('logging.level', logging.WARN, type=int))

    warning = oly_log.warning
    info = oly_log.info
    debug = oly_log.debug
    error = oly_log.error
    critical = oly_log.critical
    exception = oly_log.exception


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


def seed(seed):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


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


