import logging
import sys
import time

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
    mbs_log.setLevel(level)


def log_record(name, level, path, lno, msg, args, exc_info, func=None, sinfo=None, **kwargs):
    start = path.rfind('mlbaselines')
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


if globals().get('mbs_log') is None:
    mbs_log = make_logger('MBS')
    set_log_level(logging.DEBUG)

    warning = mbs_log.warning
    info = mbs_log.info
    debug = mbs_log.debug
    error = mbs_log.error
    critical = mbs_log.critical
    exception = mbs_log.exception


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
