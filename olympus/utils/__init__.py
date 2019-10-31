import logging
import sys
import time

from olympus.utils.arguments import task_arguments
from typing import Callable, Optional, TypeVar
from urllib.parse import urlparse

import torch

A = TypeVar('A')
R = TypeVar('R')


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


def set_log_level(level=logging.INFO):
    oly_log.setLevel(level)


def log_record(name, level, path, lno, msg, args, exc_info, func=None, sinfo=None, **kwargs):
    start = path.rfind('olympus')
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


def storage(uri):
    """Shorten the storage config from orion that is super long an super confusing
        <storage_type>:<database>:<file or address>

        legacy:pickleddb:my_data.pkl
        legacy:mongodb://user@pass:192.168.0.0:8989
    """
    storage_type, storage_uri = uri.split(':', maxsplit=1)
    arguments = parse_uri(storage_uri)
    database = arguments.get('scheme', 'pickleddb')
    database_resource = arguments.get('path', arguments.get('address'))

    return {
        'type': storage_type,
        'database': {
            'type': database,
            'host': database_resource,
        }
    }


if globals().get('oly_log') is None:
    oly_log = make_logger('OLY')
    set_log_level(logging.DEBUG)

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
