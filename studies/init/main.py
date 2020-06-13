import argparse
import os

import numpy

import yaml

from studies import (
    run as run_simul, HP_FIXED, SIMUL_FREE)
from studies import load_results as load_variance_results


IDENTITY_SIZE = 16


def env(namespace, hpo_namespace):
    return namespace + '-' + hpo_namespace.replace('_', '-')


def get_min_max_var_seeds(var_name, variance_namespace, variance_save_dir, objective):

    data = load_variance_results(variance_namespace, variance_save_dir)

    max_epoch = max(data.epoch.values)

    var_data = data.loc[dict(seed=var_name, epoch=max_epoch)]
    results = var_data[objective].values.reshape(-1)
    idx = numpy.argsort(results)
    min_index = idx[0]
    max_index = idx[-1]
    min_seed = int(var_data[var_name].values[min_index])
    max_seed = int(var_data[var_name].values[max_index])

    return min_seed, max_seed


def run(uri, database, namespace, function,
        fidelity, space, objective, var_name, variance_namespace,
        variance_save_dir,
        sample_size, extremum,
        variables, defaults, sleep_time=60, save_dir='.',
        seed=1):

    # Load variance json file 
    min_seed, max_seed = get_min_max_var_seeds(var_name, variance_namespace, variance_save_dir,
            objective)

    if extremum == 'min':
        extremum_seed = min_seed
    elif extremum == 'max':
        extremum_seed = max_seed
    else:
        raise ValueError(f'`extremum` should be either `min` or `max`: `{extremum}`')

    # Reuse simul

    # Don't consider this one as a variable
    variables.pop(var_name, None)
    # And set to this value in all experiments
    defaults[var_name] = extremum_seed
    simul_namespace = env(namespace, f'{extremum}')

    run_simul(
        uri, database, simul_namespace,
        function=function, fidelity=fidelity, space=space, objective=objective, variables=variables,
        defaults=defaults, sleep_time=sleep_time, save_dir=save_dir,
        num_replicates=sample_size, num_experiments=2, num_simuls=1, seed=1,
        rep_types=[HP_FIXED, SIMUL_FREE])


def run_from_config_file(uri, database, namespace, config_file, **kwargs):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    module = __import__('.'.join(config['function'].split('.')[:-1]), fromlist=[''])
    config['function'] = getattr(module, config['function'].split('.')[-1])
    config.update(kwargs)
    return run(uri, database, namespace, **config)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--uri', default='mongo://127.0.0.1:27017', type=str)
    parser.add_argument('--database', default='olympus', type=str)
    parser.add_argument('--namespace', default=None, type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--sleep-time', default=60, type=int)
    parser.add_argument('--variance-namespace', default=None, type=str)
    parser.add_argument('--variance-save-dir', default='.', type=str)
    parser.add_argument('--sample-size', default=200, type=int)
    parser.add_argument('--extremum', choices=['min', 'max'])
    parser.add_argument('--save-dir', default='.', type=str)
    args = parser.parse_args(args)

    namespace = args.namespace
    if namespace is None:
        namespace = '.'.join(os.path.basename(args.config).split('.')[:-1])

    run_from_config_file(
        args.uri, args.database, namespace, args.config,
        sample_size=args.sample_size, extremum=args.extremum,
        variance_namespace=args.variance_namespace,
        variance_save_dir=args.variance_save_dir,
        sleep_time=args.sleep_time,
        save_dir=args.save_dir)


if __name__ == '__main__':
    main()
