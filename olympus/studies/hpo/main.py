import argparse
from collections import defaultdict
import json
import time

import numpy

from msgqueue.backends import new_client
from sspace.space import compute_identity

import yaml

import xarray

from olympus.observers.msgtracker import METRIC_QUEUE
from olympus.hpo.parallel import make_remote_call, RESULT_QUEUE, WORK_QUEUE, WORK_ITEM, HPO_ITEM
from olympus.hpo import HPOptimizer, Fidelity
from olympus.utils.functional import flatten
from olympus.studies.searchspace.main import fetch_hpo_valid_curves, is_hpo_completed, is_registered


IDENTITY_SIZE = 16


def generate_grid_search(budget, fidelity, search_space, seeds):
    # TODO: Add limit to number of trials. If this is smaller then 3 ** dim,
    #       grid search should suggest a shuffled grid of size [budget].
    configs = []
    dim = len(search_space)
    n_points = 2
    while n_points ** dim < budget:
        n_points += 1

    config = {'name': 'grid_search', 'n_points': n_points, 'seed': 1}
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
            config['namespace'] = f'noisy-grid-search-p-{config["n_points"]}-s-{seed}'
            config.pop('uid')
            config['uid'] = compute_identity(config, IDENTITY_SIZE)

            configs.append(config)
    
    return configs


def generate_random_search(budget, fidelity, search_space, seeds):
    configs = []
    for seed in seeds:
        config = {'name': 'random_search', 'seed': seed}
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
        config = {'name': 'bayesopt', 'seed': seed}
        config['namespace'] = f'bayesopt-s-{seed}'
        config['space'] = search_space
        config['uid'] = compute_identity(config, IDENTITY_SIZE)

        configs.append(config)

    return []  # configs


generate_hpo_configs = dict(
    grid_search=generate_grid_search,
    nudged_grid_search=generate_nudged_grid_search,
    noisy_grid_search=generate_noisy_grid_search,
    random_search=generate_random_search,
    hyperband=generate_hyperband,
    bayesopt=generate_bayesopt)


def fetch_hpos_valid_curves(client, namespaces, variables, data):
    remainings = defaultdict(list)
    for hpo in namespaces.keys():
        for hpo_namespace in namespaces[hpo]:
            if is_hpo_completed(client, hpo_namespace):
                data[hpo].append(fetch_hpo_valid_curves(client, hpo_namespace, variables))
            else:
                remainings[hpo].append(hpo_namespace)

    return remainings



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


def fetch_registered(client, namespace, hpo, seed):
    registered = set()
    for message in client.monitor().messages(WORK_QUEUE, env(namespace, hpo, seed)):
        registered.add(compute_identity(message.message['kwargs'], IDENTITY_SIZE))
    return registered


def generate_hpos(seeds, hpos, budget, fidelity, search_space):

    configs = dict()
    for hpo in hpos:
        configs[hpo] = generate_hpo_configs[hpo](budget, fidelity, search_space, seeds)

    return configs


def register_test(client, namespace, remote_call):
    client.push(WORK_QUEUE, namespace, remote_call, mtype=WORK_ITEM)


def env(namespace, hpo_namespace):
    return namespace + '-' + hpo_namespace



def register_hpos(client, namespace, function, configs, defaults):
    namespaces = defaultdict(list)
    for hpo, hpo_configs in configs.items():
        for config in hpo_configs:
            hpo_namespace = env(namespace, config['namespace'])
            namespaces[hpo].append(hpo_namespace)
            if is_registered(client, hpo_namespace):
                print(f'HPO {hpo_namespace} already registered')
                continue
            register_hpo(client, hpo_namespace, function, config, defaults)

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
        hpo_namespaces = [hpo_data.attrs['namespace'] for hpo_data in hpo_datas]
        indices = numpy.argsort(hpo_namespaces)
        hpo_data = [hpo_datas[i] for i in indices]
        new_data[hpo] = xarray.combine_by_coords(hpo_data)
        new_data[hpo].coords['namespace'] = ('seed', list(sorted(hpo_namespaces)))

    return new_data


def save_results(namespace, data, save_dir):
    with open(f'{save_dir}/hpo_{namespace}.json', 'w') as f:
        f.write(json.dumps({hpo: d.to_dict() for hpo, d in data.items()}))


def load_results(namespace, save_dir):
    with open(f'{save_dir}/hpo_{namespace}.json', 'r') as f:
        data = {hpo: xarray.Dataset.from_dict(d) for hpo, d in json.loads(f.read()).items()}

    return data


def run(uri, database, namespace, function, num_experiments, budget, fidelity, space, objective,
        variables, defaults, sleep_time=60, do_full_train=False, save_dir='.'):

    # TODO: Implement all algos
    # hpos = ['grid_search', 'nudged_grid_search', 'noisy_grid_search', 'random_search',
    #         'hyperband', 'bayesopt']
    hpos = ['grid_search', 'nudged_grid_search', 'noisy_grid_search', 'random_search']

    if fidelity is None:
        fidelity = Fidelity(0, 0, name='epoch').to_dict()
        # TODO: Do we need epoch is space? Because this will confuse grid search.
        # space['epoch'] = 'uniform(0, 1)'
        # TODO: Add back when hyperband is implemented
        # hpos.remove(hpos.index('hyperband'))

    if num_experiments is None:
        num_experiments = 2
    else:
        # Divide for all hpos except for grid search and nudged grid search because they
        # only run once (not for all seeds)
        num_experiments = int(num_experiments / (len(hpos) - 2))

    client = new_client(uri, database)

    configs = generate_hpos(
        list(range(num_experiments)), hpos, budget,
        fidelity, space)
    namespaces = register_hpos(
        client, namespace, function, configs,
        dict(list(variables.items()) + list(defaults.items())))
    remainings = namespaces
    data = defaultdict(list)
    variable_names = list(sorted(variables.keys()))
    while sum(remainings.values(), []):
        remainings = fetch_hpos_valid_curves(client, remainings, variable_names, data)

        # TODO: Implement full-train part
        if do_full_train:
            configs = generate_tests(data, defaults, registered)
            new_registered_tests = register_tests(client, namespace, function, configs)

        time.sleep(sleep_time)

    # Save valid results
    data = consolidate_results(data)
    save_results(namespace, data, save_dir)

    import pdb
    pdb.set_trace()

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
    args = parser.parse_args(args)

    namespace = args.namespace
    if namespace is None:
        namespace = '.'.join(os.path.basename(args.config).split('.')[:-1])

    run_from_config_file(
        args.uri, args.database, namespace, args.config,
        num_experiments=args.num_experiments, budget=args.budget,
        sleep_time=args.sleep_time, 
        save_dir=args.save_dir)


if __name__ == '__main__':
    main()
