from argparse import Namespace, ArgumentParser


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


class NotHyperparameter(Exception):
    pass


def _insert_hyperparameter(hypers_dict, name, value):
    """Insert an hyper parameter inside the dictionary"""
    try:
        name, hp_name = name.replace('--', '').split('.', maxsplit=1)
        data = hypers_dict.get(name, {})

        try:
            data[hp_name] = float(value)
        except:
            data[hp_name] = value

        hypers_dict[name] = data

    except ValueError as e:
        raise NotHyperparameter(f'`{name} {value}` is not a valid argument') from e


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
        from olympus.utils import error
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
