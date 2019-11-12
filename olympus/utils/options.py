import os


def options(name, default, type=str):
    name = name.upper().replace('.', '_')
    return type(os.getenv(f'OLYMPUS_{name}', default))


def option(name, default, type=str):
    return options(name, default, type=type)
