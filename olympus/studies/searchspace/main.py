from collections import defaultdict
import argparse
import json
import os
import time

import yaml
from msgqueue.backends import new_client
import numpy
import xarray

from olympus.hpo import HPOptimizer
from olympus.hpo.fidelity import Fidelity
from olympus.observers.msgtracker import METRIC_QUEUE
from olympus.hpo.parallel import make_remote_call, RESULT_QUEUE, WORK_QUEUE, HPO_ITEM
from olympus.utils.functional import flatten
from olympus.studies.searchspace.plot import plot


def register_hpo(client, namespace, function, config, defaults):
    hpo = {
        'hpo': make_remote_call(HPOptimizer, **config),
        'hpo_state': None,
        'work': make_remote_call(function, **defaults),
        'experiment': namespace
    }
    return client.push(WORK_QUEUE, namespace, message=hpo, mtype=HPO_ITEM)


def is_registered(client, namespace):
    return client.db[WORK_QUEUE].count({'namespace':namespace, 'mtype': HPO_ITEM}) > 0


def is_hpo_completed(client, namespace):
    return client.monitor().unread_count(RESULT_QUEUE, namespace, mtype=HPO_ITEM) > 0


def get_hpo_work_state(client, namespace):
    messages = client.monitor().messages(WORK_QUEUE, namespace, mtype=HPO_ITEM)
    for m in messages:
        if m.mtype == HPO_ITEM:
            return m.message


def get_hpo_result_state(client, namespace):
    messages = client.monitor().unread_messages(RESULT_QUEUE, namespace, mtype=HPO_ITEM)
    for m in messages:
        if m.mtype == HPO_ITEM:
            return m.message


def get_hpo(client, namespace):
    result_state = get_hpo_result_state(client, namespace)
    if result_state is None:
        raise RuntimeError(f'No HPO for namespace {namespace} or HPO is not completed')

    args = result_state['hpo']['args']
    kwargs = result_state['hpo']['kwargs']
    remote_call = result_state['work']
    hpo = HPOptimizer(*args, **kwargs)

    hpo.load_state_dict(result_state['hpo_state'])

    return hpo, remote_call


def get_array_names(metrics):
    keys = set()
    for trial_metrics in metrics.values():
        for values in trial_metrics:
            keys |= set(values.keys())

    return keys


def fetch_metrics(client, namespace):
    metrics = defaultdict(list)
    for message in client.monitor().messages(METRIC_QUEUE, namespace):
        uid = message.message.pop('uid')
        metrics[uid].append(message.message)

    def get_epoch(item):
        return item.get('epoch', 1)

    return {name: list(sorted(values, key=get_epoch)) for name, values in metrics.items()}


def fetch_hpo_valid_curves(client, namespace, variables):
    hpo, remote_call = get_hpo(client, namespace)

    # NOTE: Tasks without epochs should have fidelity.max == 1
    epochs = hpo.hpo.fidelity.max
    flattened_space = flatten(hpo.hpo.space.serialize())
    flattened_space.pop('uid')
    flattened_space.pop(hpo.hpo.fidelity.name, None)
    params = list(sorted(flattened_space.keys()))
    seed = hpo.hpo.seed

    metrics = fetch_metrics(client, namespace)

    variables_values = {name: remote_call['kwargs'][name] for name in variables}

    for trial in hpo.trials.values():
        trial.params.update(variables_values)

    data = create_valid_curves_xarray(hpo.trials, metrics, variables, epochs, params, seed)

    data.attrs['namespace'] = namespace

    return data


def create_valid_curves_xarray(trials, metrics, variables, epochs, params, seed):
    epochs = list(range(epochs + 1))
    order = list(range(len(trials)))
    uids = [trial.uid for trial in trials.values()]
    uids_mapping = {str(trial_id): i for i, trial_id in enumerate(uids)}
    if 'epoch' in params:
        params.remove('epoch')
    noise_dimensions = variables

    h_params = xarray.DataArray(
        numpy.zeros((len(order), 1, len(params))),
        dims=['order', 'seed', 'param'],
        coords={'order': order,
                'uid': (('seed', 'order'), [uids]),
                'seed': [seed],
                'param': params,
                })

    trial_seeds = xarray.DataArray(
        numpy.zeros((len(order), 1, len(noise_dimensions))),
        dims=['order', 'seed', 'noise'],
        coords={'order': order,
                'uid': (('seed', 'order'), [uids]),
                'seed': [seed],
                'noise': noise_dimensions,
                })

    array_names = get_array_names(metrics)
    array_names.remove('epoch')
    array_names = list(sorted(array_names))

    arrays = {}
    for array_name in array_names:
        # NOTE: We set values to NaN to detect any missing metric which would be left to NaN
        arrays[array_name] = xarray.DataArray(
            numpy.zeros((len(epochs), len(order), 1)) * numpy.NaN,
            dims=['epoch', 'order', 'seed'],
            coords={'epoch': epochs,
                    'order': order,
                    'uid': (('seed', 'order'), [uids]),
                    'seed': [seed]})

    for trial_uid, trial in trials.items():
        uid_index = uids_mapping[trial_uid]

        for i, param in enumerate(params):
            h_params.loc[dict(order=uid_index, seed=seed, param=param)] = trial.params[param]

        for i, noise_dimension in enumerate(noise_dimensions):
            key = dict(order=uid_index, seed=seed, noise=noise_dimension)
            trial_seeds.loc[key] = trial.params[noise_dimension]

        trial_elapsed_time = []
        if trial_uid not in metrics:
            raise ValueError('Could not find metrics for trial {trial_uid}.')
        for epoch_stats in metrics[trial_uid]:
            epoch = epoch_stats.get('epoch', 0)

            key = dict(epoch=epoch, order=uid_index, seed=seed)
            for array_name in array_names:
                if array_name not in epoch_stats:
                    continue
                arrays[array_name].loc[key] = epoch_stats[array_name]

    data_vars = {
        array_name: (('epoch', 'order', 'seed'), array)
        for array_name, array in arrays.items()}

    coords = {'epoch': epochs, 'order': order, 'uid': (('seed', 'order'), [uids]), 'seed': [seed],
             'params': params, 'noise': noise_dimensions}
    for param in params:
        coords[param] = (('order', 'seed'), h_params.loc[dict(param=param)])
    for noise_dimension in noise_dimensions:
        coords[noise_dimension] = (('order', 'seed'), trial_seeds.loc[dict(noise=noise_dimension)])

    data = xarray.Dataset(
        data_vars=data_vars,
        coords=coords)

    return data


def save_results(namespace, data, save_dir):
    with open(f'{save_dir}/{namespace}.json', 'w') as f:
        f.write(json.dumps(data.to_dict()))


def load_results(namespace, save_dir):
    with open(f'{save_dir}/{namespace}.json', 'r') as f:
        data = xarray.Dataset.from_dict(json.loads(f.read()))

    return data


def run(uri, database, namespace, function, fidelity, space, count, variables, 
        plot_filename, objective, defaults, save_dir='.', sleep_time=60):
    if fidelity is None:
        fidelity = Fidelity(1, 1, name='epoch').to_dict()

    defaults.update(variables)

    defaults.update(variables)

    config = {
        'name': 'random_search',
        'fidelity': fidelity,
        'space': space,
        'count': count
        }

    client = new_client(uri, database)

    if not is_registered(client, namespace):
        register_hpo(client, namespace, function, config, defaults=defaults)

    while not is_hpo_completed(client, namespace):
        time.sleep(sleep_time)

    # get the result of the HPO
    print(f'HPO is done')
    data = fetch_hpo_valid_curves(client, namespace, list(sorted(variables.keys())))
    save_results(namespace, data, save_dir)

    plot(space, objective, data, plot_filename, model_seed=1)


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
    parser.add_argument('--max-trials', default=200, type=int)
    parser.add_argument('--save-dir', default='.', type=str)
    parser.add_argument('--plot-filename', default=None, type=str)

    args = parser.parse_args(args)

    namespace = args.namespace
    if namespace is None:
        namespace = '.'.join(os.path.basename(args.config).split('.')[:-1])

    if args.plot_filename is None:
        args.plot_filename = namespace + '.png'

    run_from_config_file(
        args.uri, args.database, namespace, args.config,
        sleep_time=args.sleep_time, count=args.max_trials,
        save_dir=args.save_dir, plot_filename=args.plot_filename)


if __name__ == '__main__':
    main()
