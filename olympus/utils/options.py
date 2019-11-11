import os


def options(name, default):
    name = name.upper().replace('.', '_')
    return os.getenv(f'OLYMPUS_{name}', default)


def option(name, default):
    return options(name, default)
