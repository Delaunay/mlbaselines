import argparse
from collections import defaultdict
import datetime
import json
import time
import os
import copy
import re
import pprint

import numpy

from msgqueue.backends import new_client
from sspace.space import compute_identity
from sspace import Space

import yaml

import xarray

from olympus.observers.msgtracker import METRIC_QUEUE
from olympus.hpo.parallel import (
    make_remote_call, RESULT_QUEUE, WORK_QUEUE, WORK_ITEM, HPO_ITEM, RESULT_ITEM)
from olympus.hpo.robo import build_model, build_bounds
from olympus.hpo import HPOptimizer, Fidelity
from olympus.utils.functional import flatten
from olympus.studies.searchspace.main import (
    fetch_hpo_valid_curves, is_hpo_completed, is_registered, fetch_metrics,
    create_valid_curves_xarray)
from olympus.studies.variance.main import wait, fetch_vars_stats, remaining, create_trials
from olympus.studies.hpo.main import (
    generate_hpos, register_hpos, fetch_hpo_stats, fetch_all_hpo_stats, fetch_hpos_valid_curves,
    fetch_trial_stats, print_status)


IDENTITY_SIZE = 16


def get_configs_to_replicate(configs, num_simuls):
    to_replicate = dict()
    for hpo, hpo_configs in configs.items():
        to_replicate[hpo] = []
        for i, (hpo_namespace, config) in enumerate(sorted(hpo_configs.items())):
            if i >= num_simuls:
                break
            to_replicate[hpo].append(hpo_namespace)

    return to_replicate


def reset_pool_size(configs):
    for hpo_namespace, config in configs.items():
        config['pool_size'] = None
        config.pop('uid', None)
        config['uid'] = compute_identity(config, IDENTITY_SIZE)


def randomize_seeds(configs, variables, seed, compute_id=False):
    rng = numpy.random.RandomState(seed)

    for hpo_namespace, config in configs.items():
        # config['defaults'] = copy.deepcopy(config.get('defaults', {}))
        config.setdefault('defaults', {})
        for variable in variables:
            config['defaults'][variable] = rng.randint(2**30)

        if compute_id:
            config['defaults'].pop('uid', None)
            config['defaults']['uid'] = compute_identity(config['defaults'], IDENTITY_SIZE)


def generate_biased_replicates(
        data, config, variables, objective, hpo_budget, num_replicates,
        early_stopping=True):

    space = config['space']
    # Replace all NaN is objective by float('inf')
    objectives = data[objective].fillna(float('inf'))
    # Only select first `hpo_budget` trials. The remainder are necessary for fitting
    # surrogate model only.
    objectives = objectives.sel(order=range(0, hpo_budget))
    # Find best epoch
    if early_stopping:
        objectives = numpy.minimum.accumulate(objectives.values, axis=0)
        objectives = objectives[-1, :, :].reshape(-1)
    # Use last epoch
    else:
        objectives = objectives.isel(epoch=-1).values.reshape(-1)
    assert len(objectives) == hpo_budget
    # Find best trial based on best epochs
    trial_index = numpy.argmin(objectives)
    # Take these HPs
    config = copy.deepcopy(config)
    for param_name in space:
        # NOTE: We assume all params are real, not discreate nor categorical
        config['defaults'][param_name] = float(data[param_name].sel(order=trial_index).values)

    configs = {i: copy.deepcopy(config) for i in range(num_replicates)}
    randomize_seeds(configs, variables, config['seed'], compute_id=True)

    return [configs[i]['defaults'] for i in range(num_replicates)]


def generate_simulated_replicates(simulated_fix_configs, config, variables):
    simulated_configs = {
        i: {'defaults': copy.deepcopy(c)}
        for i, c in enumerate(simulated_fix_configs)}
    # NOTE: Biased and simulated replicates have the same seeds for sources of variance
    #       This enables paired comparisons.
    randomize_seeds(simulated_configs, variables, config['seed'], compute_id=True)
    return [simulated_configs[i]['defaults'] for i in range(len(simulated_fix_configs))]


def generate_simulated_fix(data, config, variables, objective, hpo_budget, num_replicates,
                           early_stopping=True):
    # Don't forget to update uid of trial after sampling new hps.
    seeds = numpy.random.RandomState(config['seed']).randint(2**30, size=num_replicates + 1)
    space = config['space']

    X, y = convert_data_to_xy(data, space, objective, early_stopping)
    X, y = cutoff(X, y, percentile=0.85)
    model = fit_model(X, y, space, seed=seeds[0])
    configs = []
    for i, seed in enumerate(seeds[1:]):
        params = simulate_hpo(model, space, hpo_budget, seed)
        replicate = copy.deepcopy(config['defaults'])
        replicate.update(params)
        replicate.pop('uid', None)
        replicate['uid'] = compute_identity(replicate, IDENTITY_SIZE)
        configs.append(replicate)

    return configs


def get_model_init_var(variables):
    for candidate in ['init_seed', 'random_state']:
        if candidate in variables:
            return candidate

    raise RuntimeError('Could not find the variable name for model initialization')


def get_bootstrap_var(variables):
    for candidate in ['bootstrapping_seed', 'bootstrap_seed']:
        if candidate in variables:
            return candidate

    raise RuntimeError('Could not find the variable name for bootstrapping')


def limit_to_var(configs, ref_config, var):
    new_configs = []
    for config in configs:
        # Make sure we have HPs from config
        new_config = copy.deepcopy(config)
        # But update variables with default values
        new_config.update(ref_config)
        # And bring back the single var we want to vary
        new_config[var] = config[var]
        # Update corresponding uid
        new_config.pop('uid', None)
        new_config['uid'] = compute_identity(new_config, IDENTITY_SIZE)
        new_configs.append(new_config)

    return new_configs


def generate_hpo_replicates(data, config, variables, objective, hpo_budget, num_replicates,
                            early_stopping):

    replicates = dict()
    replicates['biased'] = generate_biased_replicates(
        data, config, variables, objective, hpo_budget, 
        num_replicates, early_stopping=early_stopping)

    replicates['weights_init'] = limit_to_var(
        replicates['biased'], config['defaults'], get_model_init_var(variables))
    replicates['bootstrap'] = limit_to_var(
        replicates['biased'], config['defaults'], get_bootstrap_var(variables))

    replicates['simul-fix'] = generate_simulated_fix(
        data, config, variables, objective, hpo_budget, num_replicates,
        early_stopping=early_stopping)
    replicates['simul-free'] = generate_simulated_replicates(
        replicates['simul-fix'], config, variables)

    return replicates


def generate_replicates(hpos_ready, data, variables, objective, hpo_budget, num_replicates,
                        early_stopping):
    replicates = dict()
    for hpo, configs in hpos_ready.items():
        replicates[hpo] = dict()
        for namespace, config in configs.items():
            replicates[hpo][namespace] = generate_hpo_replicates(
                data[hpo][namespace], config, variables, objective, hpo_budget, num_replicates,
                early_stopping)

    return replicates


def get_ready_configs(hpos_ready, configs, to_replicate):
    ready_configs = defaultdict(dict)
    # configs = {hpo: {config['namespace']: config for config in hpo_configs}
    #            for hpo, hpo_configs in configs.items()}
    for hpo, namespaces in hpos_ready.items():
        for namespace in namespaces:
            if namespace in to_replicate[hpo]:
                ready_configs[hpo][namespace] = configs[hpo][namespace]

    return ready_configs


def env(namespace, hpo_namespace):
    return namespace + '-' + hpo_namespace.replace('_', '-')


def consolidate_results(data_hpo, data_replicates):
    new_data = dict()
    rep_types = ['weights_init', 'bootstrap', 'biased', 'simul-fix', 'simul-free']
    for hpo, hpo_datas in data_replicates.items():
        ideal_datas = data_hpo[hpo]
        hpo_namespaces = sorted(ideal_datas.keys())
        ideal_data = [ideal_datas[namespace] for namespace in hpo_namespaces]
        new_data[hpo] = dict(ideal=xarray.combine_by_coords(ideal_data))
        new_data[hpo]['ideal'].coords['namespace'] = ('seed', hpo_namespaces)

        for replication_type in rep_types:
            hpo_namespaces = sorted(hpo_datas.keys())
            replicates_data = [hpo_datas[hpo_namespace][replication_type]
                               for hpo_namespace in hpo_namespaces]
            new_data[hpo][replication_type] = xarray.combine_by_coords(replicates_data)
            replicate_namespaces = [env(hpo_namespace, replication_type)
                                    for hpo_namespace in hpo_namespaces]
            new_data[hpo][replication_type].coords['namespace'] = ('seed', replicate_namespaces)

    return new_data


def save_results(namespace, data, save_dir):
    with open(f'{save_dir}/simul_{namespace}.json', 'w') as f:
        f.write(
            json.dumps({
                hpo: {
                    rep_type: d.to_dict()
                    for rep_type, d in reps.items()
                }
                for hpo, reps in data.items()
            })
        )


def load_results(namespace, save_dir):
    with open(f'{save_dir}/simul_{namespace}.json', 'r') as f:
        data = {
            hpo: {
                rep_type: xarray.Dataset.from_dict(d)
                for rep_type, d in reps.items()
            }
            for hpo, reps in json.loads(f.read()).items()
        }

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


def print_status_shaesthasehu(client, namespace, namespaces, hpo_stats=None):
    if hpo_stats is None:
        hpo_stats = fetch_hpo_stats(client, namespace)
    trial_stats = fetch_trial_stats(client, namespace)

    print()
    print(datetime.datetime.now())
    print((' ' * 17) + 'HPO   completed    pending     count     broken')
    for hpo, hpo_namespaces in namespaces.items():
        status = hpo_stats.get(env(namespace, hpo)[:len(namespace) + 9])
        if status is None:
            status = dict(actioned=0, read=0, count=0, error=0)
        status['pending'] = status['count'] - status['actioned']
        print(f'{hpo:>20}: {status["actioned"]:>10} {status["pending"]:>10}'
              f'{status["count"]:>10} {status["error"]:>10}')
        status = trial_stats.get(env(namespace, hpo)[:len(namespace) + 9])
        if status is None:
            status = dict(actioned=0, read=0, count=0, error=0)
        status['pending'] = status['count'] - status['actioned']
        label = 'trials'
        print(f'{label:>20}: {status["actioned"]:>10} {status["pending"]:>10}'
              f'{status["count"]:>10} {status["error"]:>10}')
        print()


def fetch_all_replicate_trials(client, namespace):
    pass


def convert_data_to_xy(data, space, objective, early_stopping=True):
    X = numpy.zeros((len(data.order), len(data.params)))

    for j, param in enumerate(sorted(data.params.values)):
        # NOTE: We assue all params are real, not discrete nor categorical
        X[:, j] = data[param].values.reshape(-1)
        if space[param].startswith('log'):
            X[:, j] = numpy.log(X[:, j])

    objectives = data[objective].fillna(float('inf'))
    if early_stopping:
        objectives = numpy.minimum.accumulate(objectives.values, axis=0)  # TODO which axis?
        y = objectives[-1, :, :].reshape(-1)
    # Use last epoch
    else:
        y = objectives.isel(epoch=-1).values.reshape(-1)

    return X, y


def convert_samples_to_x(samples, space):
    X = numpy.zeros((len(samples), len(space)))

    for i, sample in enumerate(samples):
        for j, (param_name, prior) in enumerate(sorted(space.items())):
            X[i, j] = sample[param_name]
            if prior.startswith('log'):
                X[i, j] = numpy.log(X[i, j])

    return X


def convert_x_to_samples(X, space):
    samples = []

    for i in range(X.shape[0]):
        sample = dict()
        for j, (param_name, prior) in enumerate(sorted(space.items())):
            sample[param_name] = X[i, j]
            if prior.startswith('log'):
                sample[param_name] = numpy.exp(sample[param_name])
        samples.append(sample)

    return samples


def cutoff(X, y, percentile=0.85):

    # Remove NaNs
    # idx = (1 - numpy.isnan(all_y.reshape(-1))).astype(bool)
    # all_all_x = all_x
    # all_all_y = all_y
    # all_x = all_x[idx]
    # all_y = all_y[idx]

    idx = numpy.argsort(y)[:int(percentile * X.shape[0] + 0.5)]
    
    # threshold = all_y[idx[int(numpy.ceil(cutoff * all_x.shape[0]))]]
    # threshold = all_y[idx[min(int(cutoff * all_x.shape[0]), all_x.shape[0] - 1)]]

    # idx = all_y.reshape(-1) < float(threshold)
    assert y[idx].min() == y.min(), (y[idx].min(), y.min())
    assert y[idx].max() <= y.max(), (y[idx].max(), y.max())
    ratio = y[idx].shape[0] / y.shape[0]
    assert percentile * 0.88 <= ratio and ratio <= percentile * 1.2, (percentile, ratio)

    return X[idx], y[idx]


def fit_model(X, y, space, seed=1):
    
    n_samples = 50

    # kernel = GPy.kern.Matern52(all_x.shape[1], variance=0.01, ARD=True)
    # model = GPModel_MCMC(n_samples=n_samples, kernel=kernel)
    # with all_logging_disabled(logging.ERROR):
    #     model.updateModel(all_x, all_y, None, None)

    orion_space = Space.from_dict(space).instantiate('Orion')
    lower, upper = build_bounds(orion_space)
    # TODO set the seeds
    model = build_model(lower, upper, model_type='gp_mcmc', model_seed=1, prior_seed=1)
    model.train(X, y)

    return model


def simulate_hpo(model, space, budget, seed):
    # NOTE: We hardcode for random search for now.

    # Sample points
    samples = Space.from_dict(space).sample(budget, seed=seed)
    X = convert_samples_to_x(samples, space)

    # Score points
    mean, std = model.predict(X)
    scores = mean

    # Select best
    min_index = numpy.argmin(scores)
    best_point = X[min_index]

    return convert_x_to_samples(X[min_index].reshape(1, -1), space)[0]


def fetch_all_trial_info(client, namespace):
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
            'uid': {
                '$last': '$uid'
            },
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
            '_id': '$uid',
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

    return stats


def register_hpo_replicates(client, function, hpo_namespace, configs):
    new_registered = set()
    for replicate_type, replicate_configs in configs.items():
        rep_namespace = env(hpo_namespace, replicate_type)
        registered = set(fetch_all_trial_info(client, rep_namespace).keys())
        for config in replicate_configs:
            if config['uid'] in registered:
                print(f'trial {config["uid"]} already registered')
                continue
            client.push(WORK_QUEUE, rep_namespace, make_remote_call(function, **config),
                        mtype=WORK_ITEM)
            new_registered.add(config['uid'])
            registered.add(config['uid'])

    return new_registered


def register_all_replicates(client, function, namespace, replicates):

    new_registered = set()
    for hpo, hpo_replicates in replicates.items():
        for hpo_namespace, configs in hpo_replicates.items():
            new_registered |= register_hpo_replicates(client, function, hpo_namespace, configs)

    return new_registered


def remaining(hpo_stats):
    for namespace, stats in hpo_stats.items():
        if stats.get('count', 0) == 0:
            return True
        elif stats.get('count', 0) > stats.get('actioned', 0):
            return True

    return False


def fetch_hpos_replicates(client, hpo_configs, replicate_configs, variables, space):
    hpos_ready = defaultdict(list)
    remainings = defaultdict(list)

    data_replicates = dict()
    for hpo in replicate_configs.keys():
        data_replicates[hpo] = dict()
        for hpo_namespace in replicate_configs[hpo]:
            print(f'Fetching replicates of {hpo_namespace}')
            data_replicates[hpo][hpo_namespace] = dict()
            for simul_type in ['weights_init', 'bootstrap', 'biased', 'simul-fix', 'simul-free']:
                print(f'    of type {simul_type}')

                simul_configs = replicate_configs[hpo][hpo_namespace][simul_type]

                hpo_replicates = fetch_hpo_replicates(
                    client, env(hpo_namespace, simul_type), simul_configs, variables, space,
                    hpo_configs[hpo][hpo_namespace]['seed'])
                if not hpo_replicates:
                    raise RuntimeError(
                        f'{hpo_namespace} replicates are completed but no metrics are available!?')

                data_replicates[hpo][hpo_namespace][simul_type] = hpo_replicates

    return data_replicates


def fetch_hpo_replicates(client, namespace, rep_configs, variables, space, seed):

    epoch = rep_configs[0].get('epoch', 1)

    params = {key: 'should-be-overwritten-in-create_trials' for key in space.keys()}

    metrics = fetch_metrics(client, namespace)
    trials = create_trials(rep_configs, params, metrics)
    data = create_valid_curves_xarray(
        trials, metrics, variables, epoch,
        list(sorted(params.keys())), seed)

    return data


def run(uri, database, namespace, function, num_experiments, num_simuls,
        fidelity, space, objective, variables, defaults,
        num_replicates=None,
        sleep_time=60, do_full_train=False, save_dir='.', seed=1,
        register=True):

    hpo_budget = 100
    surrogate_budget = 200

    if num_replicates is None:
        num_replicates = num_experiments

    # We use 200 trials to fit the surrogate models (surrogate_budget is 200)
    # but we only need 100 for the ideal (hpo_budget is 100)
    # therefore, since num_simuls is at least half smaller than number of 
    # replicates, we can run only (num_replicates / 2) hpo runs and use
    # first half and second 100 half as 2 separe ideal runs.
    # This is possible since we are using random search.

    assert (num_experiments % 2) == 0
    assert num_simuls <= (num_experiments / 2)

    num_ideal = num_experiments // 2

    hpo = 'random_search'

    # TODO
    # for each repetition, vary all sources of variations
    # when one hpo is done, create all biased and simulations

    if fidelity is None:
        fidelity = Fidelity(1, 1, name='epoch').to_dict()

    client = new_client(uri, database)

    configs = generate_hpos(
        list(range(num_ideal)), [hpo], surrogate_budget,
        fidelity, space, namespace, defaults)

    to_replicate = get_configs_to_replicate(configs, num_simuls)

    reset_pool_size(configs['random_search'])
    randomize_seeds(configs['random_search'], variables, seed)

    variable_names = list(sorted(variables.keys()))

    hpo_stats = fetch_all_hpo_stats(client, namespace)

    namespaces = register_hpos(
        client, namespace, function, configs, defaults, hpo_stats,
        register=register)
    remainings = namespaces

    data_hpo = defaultdict(dict)
    all_replicates = dict(random_search=dict())
    while sum(remainings.values(), []):
        print_status(client, namespace, namespaces)
        hpos_ready, remainings = fetch_hpos_valid_curves(
            client, remainings, variable_names, data_hpo)

        ready_configs = get_ready_configs(hpos_ready, configs, to_replicate)

        replicates = generate_replicates(
            ready_configs, data_hpo, variables, objective, hpo_budget, num_replicates,
            early_stopping=False)
        if register:
            registered_replicates = register_all_replicates(client, function, namespace, replicates)

        if replicates.get('random_search'):
            all_replicates['random_search'].update(replicates['random_search'])
        if sum(remainings.values(), []) and not registered_replicates:
            time.sleep(sleep_time)

    wait(client, namespace, sleep=sleep_time)

    data_replicates = fetch_hpos_replicates(
        client, configs, all_replicates, variable_names, space)

    # Save valid results
    data = consolidate_results(data_hpo, data_replicates)
    save_results(namespace, data, save_dir)

    # {
    #     'hpo1': {
    #         'ideal': ['real hpo stuff'], epoch, order, seed
    #         'biased': ['variance like'],  epoch, replicate, hpo_seed
    #         'simul-fix': ['variance like'],
    #         'simul-free': ['variance like'],
    #     }
    #     'hpo2':{
    #     }
    # }


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
    parser.add_argument('--num-replicates', default=200, type=int)
    parser.add_argument('--num-simuls', default=50, type=int)
    parser.add_argument('--save-dir', default='.', type=str)
    parser.add_argument('--monitor-only', action='store_true')
    args = parser.parse_args(args)

    namespace = args.namespace
    if namespace is None:
        namespace = '.'.join(os.path.basename(args.config).split('.')[:-1])

    run_from_config_file(
        args.uri, args.database, namespace, args.config,
        num_experiments=args.num_replicates, num_simuls=args.num_simuls,
        sleep_time=args.sleep_time,
        save_dir=args.save_dir,
        register=not args.monitor_only)


if __name__ == '__main__':
    main()
