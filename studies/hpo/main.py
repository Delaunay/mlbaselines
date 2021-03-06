import argparse
from collections import defaultdict
import datetime
import json
import time
import os
import copy
import re

import numpy

from msgqueue.backends import new_client
from sspace.space import compute_identity

import yaml

import xarray

from olympus.hpo.parallel import (
    make_remote_call, RESULT_QUEUE, WORK_QUEUE, WORK_ITEM, HPO_ITEM, RESULT_ITEM)
from olympus.hpo import HPOptimizer, Fidelity
from studies import fetch_hpo_valid_curves, is_hpo_completed

IDENTITY_SIZE = 16


def generate_grid_search(budget, fidelity, search_space, seeds):
    configs = []
    dim = len(search_space)
    n_points = 2
    while n_points ** dim < budget:
        n_points += 1

    config = {'name': 'grid_search', 'n_points': n_points, 'seed': 1, 'pool_size': 20}
    config['namespace'] = f'grid-search-p-{n_points}'
    config['space'] = search_space
    config['fidelity'] = fidelity
    config['uid'] = compute_identity(config, IDENTITY_SIZE)

    configs.append(config)

    return configs


def generate_nudged_grid_search(budget, fidelity, search_space, seeds):
    configs = generate_grid_search(budget, fidelity, search_space, [])
    config = configs[0]

    config['namespace'] = f'grid-search-nudged-p-{config["n_points"]}'
    config['nudge'] = 0.5
    config['uid'] = compute_identity(config, IDENTITY_SIZE)

    return [config]


def generate_noisy_grid_search(budget, fidelity, search_space, seeds):
    configs = []
    for seed in seeds:
        seed_configs = generate_grid_search(budget, fidelity, search_space, [])
        for config in seed_configs:
            config['name'] = 'noisy_grid_search'
            config['seed'] = seed
            config['count'] = budget
            config['namespace'] = f'noisy-grid-search-p-{config["n_points"]}-s-{seed}'
            config.pop('uid')
            config['uid'] = compute_identity(config, IDENTITY_SIZE)

            configs.append(config)
    
    return configs


def generate_random_search(budget, fidelity, search_space, seeds):
    configs = []
    for seed in seeds:
        config = {'name': 'random_search', 'seed': seed, 'pool_size': 20}
        config['namespace'] = f'random-search-s-{seed}'
        config['count'] = budget
        config['fidelity'] = fidelity
        config['space'] = search_space
        config['uid'] = compute_identity(config, IDENTITY_SIZE)

        configs.append(config)

    return configs


def generate_hyperband(budget, fidelity, search_space, seeds):
    configs = []
    # TODO: Compute budget based on cumulative number of epochs.
    #       Let infinite repetitions and stop when reaching corresponding budget.
    for seed in seeds:
        config = {'name': 'hyperband', 'seed': seed}
        config['uid'] = compute_identity(config, IDENTITY_SIZE)
        config['namespace'] = f'hyperband-s-{seed}'

        configs.append(config)

    return []  # configs


def generate_bayesopt(budget, fidelity, search_space, seeds):
    configs = []
    for seed in seeds:
        rng = numpy.random.RandomState(seed)
        config = {
            'name': 'robo', 
            'model_type': 'gp_mcmc',
            'maximizer': 'random',
            'n_init': 20,
            'count': budget,
            'acquisition_func': 'log_ei',
            'model_seed': rng.randint(2**30),
            'prior_seed': rng.randint(2**30),
            'init_seed': rng.randint(2**30),
            'maximizer_seed': rng.randint(2**30)
        }
        config['fidelity'] = fidelity
        config['namespace'] = f'bayesopt-s-{seed}'
        config['space'] = search_space
        config['uid'] = compute_identity(config, IDENTITY_SIZE)

        configs.append(config)

    return configs


generate_hpo_configs = dict(
    grid_search=generate_grid_search,
    nudged_grid_search=generate_nudged_grid_search,
    noisy_grid_search=generate_noisy_grid_search,
    random_search=generate_random_search,
    hyperband=generate_hyperband,
    bayesopt=generate_bayesopt)


def fetch_hpos_valid_curves(client, namespaces, variables, data, partial=False):
    hpos_ready = defaultdict(list)
    remainings = defaultdict(list)
    fetched_one = False
    for hpo in namespaces.keys():
        for hpo_namespace in namespaces[hpo]:
            if (is_hpo_completed(client, hpo_namespace) and not fetched_one) or partial:
                print(f'Fetching results of {hpo_namespace}')

                hpo_data = fetch_hpo_valid_curves(
                    client, hpo_namespace, variables, partial=partial)
                fetched_one = True
                if hpo_data:
                    data[hpo][hpo_namespace] = hpo_data
                    hpos_ready[hpo].append(hpo_namespace)
                elif partial:
                    print(f'No metrics available for {hpo_namespace}')
                else:
                    raise RuntimeError(
                        f'{hpo_namespace} is completed but no metrics are available!?')
            else:
                remainings[hpo].append(hpo_namespace)

    return hpos_ready, remainings


def generate_grid_search_tests(client, budget, namespace):
    uids = fetch_registered_tests(client, env(namespace, 'tests'))

    for trial in hpo.trials:
        if trial.uid in uids:
            print(f'Trial {trial.uid} already registered.')
            continue

        trial_remote_call = copy.deepcopy(remote_call)
        trial_remote_call['kwargs']['hpo_done'] = True
        register_test(client, env(namespace, 'tests'), trial_remote_call)


def generate_regret_tests(client, budget, namespace):
    hpo, remote_call = get_hpo(client, namespace)

    uids = fetch_registered_tests(client, env(namespace, 'tests'))

    best_objective = float('inf')
    for trial in hpo.trials:
        if trial.objectives[-1] >= best_objective:
            continue
        best_objective = trial.objectives[-1]
        if trial.uid in uids:
            print(f'Trial {trial.uid} already registered.')
            continue
        trial_remote_call = copy.deepcopy(remote_call)
        trial_remote_call['kwargs'].update(trial.params)
        trial_remote_call['kwargs']['hpo_done'] = True
        register_test(client, env(namespace, 'tests'), trial_remote_call)


def generate_random_search_tests(client, budget, namespace):
    generate_regret_tests(client, budget, namespace)


def generate_bayesopt_tests(budget, fidelity, search_space, seeds):
    generate_regret_tests(client, budget, namespace)


def generate_hyperband_tests(budget, fidelity, search_space, seeds):
    hpo, remote_call = get_hpo(client, namespace)

    uids = fetch_registered_tests(client, env(namespace, 'tests'))

    epochs_per_trial = hpo.fidelity.max

    best_trial = None
    best_objective = float('inf')
    epochs_used = 0
    for trial in hpo.trials:
        if trial.fidelity + epochs_used <= epochs_per_trial:
            epochs_used += trial.params[hpo.fidelity.name]
        else:
            epochs_used = (epochs_used + trial.fidelity) % epochs_per_trial
            if best_trial.uid in uids:
                print(f'Trial {trial.uid} already registered.')
                continue
            trial_remote_call = copy.deepcopy(remote_call)
            trial_remote_call['kwargs'].update(best_trial)
            trial_remote_call['kwargs']['hpo_done'] = True
            register_test(client, env(namespace, 'tests'), trial_remote_call)

        if trial.objectives[-1] < best_objective:
            best_trial = trial
            best_objective = trial.objectives[-1]


def fetch_hpo_stats(client, namespace):
    length = len(namespace) + 9
    query = {
        'namespace': {'$regex': re.compile(f"^{namespace}", re.IGNORECASE)},
        'mtype': HPO_ITEM}
    stats = client.db[WORK_QUEUE].aggregate([
        {'$match': query},
        {'$project': {
            'namespace': 1,
            'sub_namespace': {'$substr': ['$namespace', 0, length]},
            'read': 1,
            'mtype': 1,
            'actioned': 1,
            'heartbeat': 1,
            'error': 1,
            'retry': 1,
        }},
        {'$group': {
            '_id': '$namespace',
            'sub_namespace': {
                '$last': '$sub_namespace'
            },
            'read': {
                '$last': '$read'
            },
            'actioned': {
                '$last': '$actioned'
            },
            'error': {
                '$sum': {'$ifNull': [0, 1]}
            },
            'retry': {
                '$sum': '$retry'
            }
        }},
        {'$group': {
            '_id': '$sub_namespace',
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

    results_counts = client.db[RESULT_QUEUE].aggregate([
        {'$match': query},
        {'$project': {
            'sub_namespace': {'$substr': ['$namespace', 0, length]},
            'mtype': 1,
            'count': 1,
        }},
        {'$group': {
            '_id': '$sub_namespace',
            'count': {
                '$sum': 1
            },
        }}
    ])

    results_counts = {doc['_id']: doc for doc in results_counts}

    return stats


def fetch_all_hpo_stats(client, namespace):
    query = {
        'namespace': {'$regex': re.compile(f"^{namespace}", re.IGNORECASE)},
        'mtype': HPO_ITEM}
    stats = client.db[WORK_QUEUE].aggregate([
        {'$match': query},
        {'$project': {
            'namespace': 1,
            'read': 1,
            'mtype': 1,
            'actioned': 1,
            'heartbeat': 1,
            'error': 1,
            'retry': 1,
        }},
        {'$group': {
            '_id': '$namespace',
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
    ])
    stats = {doc['_id']: doc for doc in stats}
    return stats


def fetch_trial_stats(client, namespace):
    length = len(namespace) + 9
    query = {
        'namespace': {'$regex': re.compile(f"^{namespace}", re.IGNORECASE)},
        'mtype': WORK_ITEM}
    stats = client.db[WORK_QUEUE].aggregate([
        {'$match': query},
        {'$project': {
            'namespace': 1,
            'sub_namespace': {'$substr': ['$namespace', 0, length]},
            'uid': '$message.kwargs.uid',
            'read': 1,
            'mtype': 1,
            'actioned': 1,
            'heartbeat': 1,
            'error': 1,
            'retry': 1
        }},
        {'$group': {
            '_id': {
                'uid': '$uid',
                'namespace': '$namespace'
            },
            'sub_namespace': {
                '$last': '$sub_namespace'
            },
            'read': {
                '$last': '$read'
            },
            'actioned': {
                '$last': '$actioned'
            },
            'error': {
                '$sum': {'$ifNull': [0, 1]}
            },
            'retry': {
                '$sum': '$retry'
            }
        }},
        {'$group': {
            '_id': '$sub_namespace',
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
                '$max': '$retry'
            },
            'count': {
                '$sum': 1
            }
        }}
    ])
    stats = {doc['_id']: doc for doc in stats}

    query['mtype'] = RESULT_ITEM
    results_counts = client.db[RESULT_QUEUE].aggregate([
        {'$match': query},
        {'$project': {
            'sub_namespace': {'$substr': ['$namespace', 0, length]},
            'mtype': 1,
            'count': 1,
        }},
        {'$group': {
            '_id': '$sub_namespace',
            'count': {
                '$sum': 1
            },
        }}
    ])
    results_counts = {doc['_id']: doc for doc in results_counts}

    return stats


def fetch_all_trial_stats(client, namespace):
    length = len(namespace) + 9
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
        }},
        {'$group': {
            '_id': {
                'uid': '$uid',
                'namespace': '$namespace'
            },
            'namespace': '$namespace',
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



def fetch_registered(client, namespace, hpo, seed):
    registered = set()


    for message in client.monitor().messages(WORK_QUEUE, env(namespace, hpo, seed), mtype=HPO_ITEM):
        registered.add(compute_identity(message.message['kwargs'], IDENTITY_SIZE))
    return registered


def generate_hpos(seeds, hpos, budget, fidelity, search_space, namespace, defaults):

    configs = dict()
    for hpo in hpos:
        configs[hpo] = dict()
        hpo_configs = generate_hpo_configs[hpo](budget, fidelity, search_space, seeds)
        for config in hpo_configs:
            config['namespace'] = env(namespace, config['namespace'])
            config['defaults'] = copy.deepcopy(defaults)
            uid = config.pop('uid')
            config['uid'] = compute_identity(config, IDENTITY_SIZE)
            configs[hpo][config['namespace']] = config

    return configs


def register_test(client, namespace, remote_call):
    client.push(WORK_QUEUE, namespace, remote_call, mtype=WORK_ITEM)


def env(namespace, hpo_namespace):
    return namespace + '-' + hpo_namespace.replace('_', '-')


def register_hpos(client, namespace, function, configs, defaults, stats, register=True):
    namespaces = defaultdict(list)
    for hpo, hpo_configs in configs.items():
        for hpo_namespace, config in hpo_configs.items():
            namespaces[hpo].append(hpo_namespace)
            # TODO: make is_registered more efficient
            if hpo_namespace in stats and register:
                print(f'HPO {hpo_namespace} already registered')
                continue
            if register:
                print(f'Registering HPO {hpo_namespace}')
                hpo_defaults = config.get('defaults', {})
                hpo_defaults.update(defaults)
                register_hpo(client, hpo_namespace, function, config, hpo_defaults)

    return namespaces


def register_hpo(client, namespace, function, config, defaults):
    hpo = {
        'hpo': make_remote_call(HPOptimizer, **config),
        'hpo_state': None,
        'work': make_remote_call(function, **defaults),
        'experiment': namespace
    }
    return client.push(WORK_QUEUE, namespace, message=hpo, mtype=HPO_ITEM)


def generate_hpo_tests(client, function, namespace, hpo_namespace, defaults):
    # TODO:
    # Fetch all results sequentially
    # Take the best ones (sequentially for the HPO regret curve), and only return them to run
    # on test

    # For hyperband, take into account the number or epochs to compute the equivalent number
    # of trials.

    # For grid search, shuffle the ordering to avoid the effect of the grid.
    return


def generate_tests(data, defaults, registered):
    configs = defaultdict(dict)
    for hpo, hpo_datas in data.items():
        for hpo_data in hpo_datas:
            if env(hpo_data.attrs['namespace'], 'test') in registered.get(hpo, {}):
                continue
            hpo_configs = generate_hpo_tests[hpo](hpo_data, defaults)
            for config in hpo_configs:
                config['hpo_done'] = True
            configs[hpo][hpo_data] = hpo_configs

    return configs


def register_tests(client, namespace, function, configs):

    new_registered = defaultdict(lambda: defaultdict(set))
    for hpo, hpo_runs in configs.items():
        for hpo_namespace, test_configs in hpo_runs:
            registered = fetch_registered(client, [hpo_namespace])
            for config in configs:
                if config['uid'] in registered:
                    print(f'trial {config["uid"]} already registered')
                    continue
                client.push(WORK_QUEUE, env(hpo_namespace, 'test'),
                            make_remote_call(function, **config),
                            mtype=WORK_ITEM)
                new_registered[hpo][hpo_namespace].add(config['uid'])
                registered.add(config['uid'])

    return new_registered


def consolidate_results(data):
    new_data = dict()
    for hpo, hpo_datas in data.items():
        hpo_namespaces = sorted(hpo_datas.keys())
        hpo_data = [hpo_datas[namespace] for namespace in hpo_namespaces]
        new_data[hpo] = xarray.combine_by_coords(hpo_data)
        new_data[hpo].coords['namespace'] = ('seed', hpo_namespaces)

    return new_data


def save_results(namespace, data, save_dir):
    with open(f'{save_dir}/hpo_{namespace}.json', 'w') as f:
        f.write(json.dumps({hpo: d.to_dict() for hpo, d in data.items()}))


def load_results(namespace, save_dir):
    with open(f'{save_dir}/hpo_{namespace}.json', 'r') as f:
        data = {hpo: xarray.Dataset.from_dict(d) for hpo, d in json.loads(f.read()).items()}

    return data


def get_status(client, namespace):
    messages = client.monitor().messages(WORK_QUEUE, namespace, mtype=HPO_ITEM)
    hpo_state = None
    for m in messages:
        if m.mtype == HPO_ITEM:
            hpo_state = m.message
    assert hpo_state is not None
    args = hpo_state['hpo']['args']
    kwargs = hpo_state['hpo']['kwargs']
    hpo = HPOptimizer(*args, **hpo_state['hpo']['kwargs'])
    if hpo_state['hpo_state']:
        hpo.load_state_dict(hpo_state['hpo_state'])

    state = dict(completed=0, broken=0, pending=0)
    if hpo.is_done():
        state['status'] = 'completed'
    else:
        state['status'] = 'pending'

    for uid, trial in hpo.trials.items():
        if trial.objective:
            state['completed'] += 1
        else:
            state['pending'] += 1

    # TODO: We don't detect any broken trial so far.

    state['missing'] = hpo.hpo.count - len(hpo.trials)

    return state


def print_status(client, namespace, namespaces, hpo_stats=None):
    if hpo_stats is None:
        hpo_stats = fetch_hpo_stats(client, namespace)

    trial_stats = fetch_trial_stats(client, namespace)

    print()
    print(datetime.datetime.now())
    print((' ' * 17) + 'HPO   completed    running    pending     count     broken       retry')
    for hpo, hpo_namespaces in namespaces.items():
        status = hpo_stats.get(env(namespace, hpo)[:len(namespace) + 9])
        if status is None:
            status = dict(actioned=0, read=0, count=0, error=0)
        status['pending'] = status['count'] - status['read']
        status['running'] = status['read'] - status['actioned']
        print(f'{hpo:>20}: {status["actioned"]:>10} {status["running"]:>10} {status["pending"]:>10}'
              f'{status["count"]:>10} {status["error"]:>10}')

        status = trial_stats.get(env(namespace, hpo)[:len(namespace) + 9])
        if status is None:
            status = dict(actioned=0, read=0, count=0, error=0, retry=0)
        status['pending'] = status['count'] - status['read']
        status['running'] = status['read'] - status['actioned']
        label = 'trials'
        print(f'{label:>20}: {status["actioned"]:>10} {status["running"]:>10} {status["pending"]:>10}'
              f'{status["count"]:>10} {status["error"]:>10} {status["retry"]:>10}')
        print()


def run(uri, database, namespace, function, num_experiments, budget, fidelity, space, objective,
        variables, defaults, sleep_time=60, do_full_train=False, save_dir='.', partial=False,
        register=True):

    # TODO: Add hyperband
    hpos = ['grid_search', 'nudged_grid_search', 'noisy_grid_search', 'random_search',
            'bayesopt']

    if fidelity is None:
        fidelity = Fidelity(1, 1, name='epoch').to_dict()

    # TODO: Add back when hyperband is implemented
    # if fidelity['min'] == fidelity['max']:
    #     hpos.remove(hpos.index('hyperband'))

    if num_experiments is None:
        num_experiments = 2

    client = new_client(uri, database)

    hpo_stats = fetch_all_hpo_stats(client, namespace)

    configs = generate_hpos(
        list(range(num_experiments)), hpos, budget,
        fidelity, space, namespace, defaults)

    variable_names = list(sorted(variables.keys()))

    if partial:
        namespaces = defaultdict(list)
        for hpo, hpo_configs in configs.items():
            for hpo_namespace, config in hpo_configs.items():
                namespaces[hpo].append(hpo_namespace)

        data = defaultdict(dict)
        fetch_hpos_valid_curves(client, namespaces, variable_names, data, partial=True)

        data = consolidate_results(data)
        save_results(namespace, data, save_dir)

        return

    namespaces = register_hpos(
        client, namespace, function, configs,
        dict(list(variables.items()) + list(defaults.items())),
        hpo_stats, register)
    remainings = namespaces

    print_status(client, namespace, namespaces)
    data = defaultdict(dict)
    while sum(remainings.values(), []):
        hpos_ready, remainings = fetch_hpos_valid_curves(client, remainings, variable_names, data)

        # TODO: Implement full-train part
        if do_full_train:
            configs = generate_tests(data, defaults, registered)
            new_registered_tests = register_tests(client, namespace, function, configs)

        if not sum(hpos_ready.values(), []):
            print_status(client, namespace, namespaces)
            time.sleep(sleep_time)

    # Save valid results
    data = consolidate_results(data)
    save_results(namespace, data, save_dir)

    if not do_full_train:
        return

    # TODO: Implement full-train part
    wait(completed)  # take the sum of all hpo_namespaces

    # NOTE & TODO: This should follow the same format as valid results, but we need to
    #              make sure the mapping in order of trials is the same.
    data = fetch_results(client, namespace, namespaces)

    # Save test results
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
    parser.add_argument('--budget', default=200, type=int)
    parser.add_argument('--save-dir', default='.', type=str)
    parser.add_argument('--monitor-only', action='store_true')
    parser.add_argument(
        '--fetch-partial', action='store_true',
        help='Do not run anything, just fetch partial results and save.')
    args = parser.parse_args(args)

    namespace = args.namespace
    if namespace is None:
        namespace = '.'.join(os.path.basename(args.config).split('.')[:-1])

    run_from_config_file(
        args.uri, args.database, namespace, args.config,
        num_experiments=args.num_experiments, budget=args.budget,
        sleep_time=args.sleep_time, 
        save_dir=args.save_dir, partial=args.fetch_partial,
        register=not args.monitor_only)


if __name__ == '__main__':
    main()
