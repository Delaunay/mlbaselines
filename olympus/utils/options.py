import os
import json

from olympus.utils.functional import flatten


_options = {}


def load_configuration(file_name):
    global _options

    options = json.load(open(file_name, 'r'))
    _options = flatten(options)


def set_option(name, value):
    global _options
    _options[name] = value


def options(name, default, type=str):
    """Look for an option locally and using the environment variables
    Environment variables are use as the ultimate overrides
    """

    env_name = name.upper().replace('.', '_')
    value = os.getenv(f'OLYMPUS_{env_name}', None)

    if not value:
        return type(_options.get(name, default))

    return type(value)


def option(name, default, type=str):
    return options(name, default, type=type)

