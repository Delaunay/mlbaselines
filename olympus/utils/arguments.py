import argparse


def task_arguments(name, description, subparser=None):
    if subparser:
        return subparser.add_parser(name, help=description)
    else:
        return argparse.ArgumentParser(description=description, prog=f'olympus {name}')

