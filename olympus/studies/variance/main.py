import argparse
import datetime
import re
from collections import OrderedDict
import copy
import json
import numpy
import time
import os
import yaml

import xarray

from sspace.space import compute_identity
from msgqueue.backends import new_client

from olympus.hpo.optimizer import Trial
from olympus.hpo.parallel import make_remote_call, RESULT_QUEUE, WORK_QUEUE, WORK_ITEM, HPO_ITEM
from olympus.studies.searchspace.main import fetch_metrics, create_valid_curves_xarray


IDENTITY_SIZE = 16


def env(namespace, median):
    return namespace + '-' + median


def get_medians(data, medians, objective):
    median_seeds = dict()
    max_epoch = max(data.epoch.values)

    for median in medians:
        median_data = data.loc[dict(seed=median, epoch=max_epoch)]
        results = median_data[objective].values.reshape(-1)
        idx = numpy.argsort(results)
        median_index = idx[(len(results) - 1) // 2]
        median_seeds[median] = int(median_data[median].values[median_index])

    return median_seeds


def fetch_registered(client, namespaces):
    registered = set()
    for namespace in namespaces:
        for message in client.monitor().messages(WORK_QUEUE, namespace):
            registered.add(message.message['kwargs']['uid'])
    return registered


def generate(seeds, variables, defaults):
    configs = dict()
    for variable in variables:
        configs[variable] = []
        for seed in seeds:
            kwargs = copy.copy(defaults)
            kwargs[variable] = int(seed)
            uid = compute_identity(kwargs, IDENTITY_SIZE)
            kwargs['uid'] = uid
            configs[variable].append(kwargs)

    return configs 


def register(client, function, namespace, variables):

    registered = fetch_registered(
        client,
        [env(namespace, variable) for variable in variables.keys()])
    new_registered = set()
    for variable, configs in variables.items():
        for config in configs:
            if config['uid'] in registered:
                print(f'trial {config["uid"]} already registered')
                continue
            client.push(WORK_QUEUE, env(namespace, variable), make_remote_call(function, **config),
                        mtype=WORK_ITEM)
            new_registered.add(config['uid'])
            registered.add(config['uid'])

    return new_registered


def wait(client, namespace, sleep=60):
    trial_stats = fetch_vars_stats(client, namespace)
    print_status(trial_stats)
    while remaining(trial_stats):
        time.sleep(sleep)
        trial_stats = fetch_vars_stats(client, namespace)
        print_status(trial_stats)
    return


# def remaining(client, namespace, variables):
#     for variable in variables:
#         print(client.monitor().messages(WORK_QUEUE, env(namespace, variable))[-1].to_dict())
#         if (client.monitor().unactioned_count(WORK_QUEUE, env(namespace, variable)) +
#             client.monitor().unread_count(WORK_QUEUE, env(namespace, variable))):
#             return True
#     return False


def remaining(status):
    return any(status[key]['actioned'] < status[key]['count'] for key in status.keys())


def fetch_vars_stats(client, namespace):
    length = len(namespace) + 6
    query = {
        'namespace': {'$regex': re.compile(f"^{namespace}", re.IGNORECASE)},
        'mtype': WORK_ITEM}
    stats = client.db[WORK_QUEUE].aggregate([
        {'$match': query},
        {'$project': {
            'namespace': 1,
            'uid': '$message.kwargs.uid',
            'read': 1,
            'mtype': 1,
            'actioned': 1,
            'heartbeat': 1,
            'error': 1,
            'retry': 1,
        }},
        {'$group': {
            '_id': '$uid',
            'namespace': {
                '$last': '$namespace'
            },
            'read': {
                '$last': '$read'
            },
            'actioned': {
                '$last': '$actioned'
            },
            'error': {
                '$sum': {'$cond': [{'$eq':["$error", True]}, 1, 0]}
            },
            'retry': {
                '$sum': '$retry'
            }
        }},
        {'$group': {
            '_id': '$namespace',
            'read': {
                '$sum': {'$cond': [{'$eq':["$read", True]}, 1, 0]}
            },
            'actioned': {
                '$sum': {'$cond': [{'$eq':["$actioned", True]}, 1, 0]}
            },
            'error': {
                '$sum': '$error'
            },
            'retry': {
                '$sum': '$retry'
            },
            'count': {
                '$sum': 1
            }
        }}
    ])
    stats = {doc['_id']: doc for doc in stats}

    return stats


def print_status(trial_stats):

    print()
    print(datetime.datetime.now())
    print((' ' * 32) + 'variable   completed    pending     count     broken')
    for namespace, status in trial_stats.items():
        # status = dict(
        #     completed=0, broken=0, pending=0,
        #     trials=dict(completed=0, broken=0, pending=0, missing=0))
        # for hpo_namespace in hpo_namespaces:
        #     hpo_status = get_status(client, hpo_namespace)
        #     status[hpo_status['status']] += 1
        #     for key in ['completed', 'broken', 'pending', 'missing']:
        #         status['trials'][key] += hpo_status[key]
        status['pending'] = status['count'] - status['actioned']
        print(f'{namespace:>40}: {status["actioned"]:>10} {status["pending"]:>10}'
              f'{status["count"]:>10} {status["error"]:>10}')
        print()


def fetch_all_metrics(client, namespace, variables):
    metrics = {}
    for variable in variables:
        v_metrics = fetch_metrics(client, env(namespace, variable))
        msg = 'There was duplicates between variable experiments.'
        assert len(set(v_metrics.keys()) & set(metrics.keys())) == 0, msg
        metrics.update(v_metrics)

    return metrics


def fetch_results(client, namespace, configs, medians, params, defaults):

    variables = list(sorted(configs.keys()))

    metrics = fetch_all_metrics(client, namespace, variables)

    epoch = defaults.get('epoch', 1)

    arrays = []
    trial_stats = fetch_vars_stats(client, namespace)
    if remaining(trial_stats):
        raise RuntimeError('Not all trials are completed')
    for variable in configs.keys():
        trials = create_trials(configs[variable], params, metrics)
        arrays.append(
            create_valid_curves_xarray(
                trials, metrics, variables, epoch,
                list(sorted(params.keys())), variable))

    data = xarray.combine_by_coords(arrays)
    data.attrs['medians'] = medians
    data.coords['namespaces'] = (
        ('seed', ), [env(namespace, v) for v in sorted(configs.keys())])

    return data


def create_trials(configs, params, metrics):
    trials = OrderedDict()
    for config in configs:
        uid = config['uid']
        if uid not in metrics:
            raise RuntimeError(
                'Nothing found in result queue for trial {uid}. Is it really completed?')
        params = copy.deepcopy(params)
        params.update(config)
        params['uid'] = uid
        # NOTE: We don't need objectives
        trials[uid] = Trial(params)

    return trials


def save_results(namespace, data, save_dir):
    with open(f'{save_dir}/variance_{namespace}.json', 'w') as f:
        f.write(json.dumps(data.to_dict()))


def load_results(namespace, save_dir):
    with open(f'{save_dir}/variance_{namespace}.json', 'r') as f:
        data = xarray.Dataset.from_dict(json.loads(f.read()))

    return data


def run(uri, database, namespace, function, objective, medians, defaults, variables, params,
        num_experiments, sleep_time=60, save_dir='.'):
    if num_experiments is None:
        num_experiments = 20

    client = new_client(uri, database)

    defaults.update(dict(list(variables.items()) + list(params.items())))

    configs = generate(range(num_experiments), medians, defaults)
    register(client, function, namespace, configs)

    wait(client, namespace, sleep=sleep_time)

    data = fetch_results(client, namespace, configs, medians, params, defaults)
    defaults.update(get_medians(data, medians, objective))
    new_configs = generate(range(num_experiments), variables, defaults)
    register(client, function, namespace, new_configs)

    wait(client, namespace, sleep=5)

    configs.update(new_configs)
    data = fetch_results(client, namespace, configs, variables, params, defaults)

    save_results(namespace, data, save_dir)


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
    parser.add_argument('--num-experiments', default=200, type=int)
    parser.add_argument('--save-dir', default='.', type=str)

    args = parser.parse_args(args)

    namespace = args.namespace
    if namespace is None:
        namespace = '.'.join(os.path.basename(args.config).split('.')[:-1])

    run_from_config_file(
        args.uri, args.database, namespace, args.config,
        sleep_time=args.sleep_time, num_experiments=args.num_experiments,
        save_dir=args.save_dir)


if __name__ == '__main__':
    main()
