import copy
from collections import defaultdict, OrderedDict

import numpy
import xarray

import pymongo

from sspace.space import compute_identity
from sspace import Space

from msgqueue.backends import new_client

from olympus.hpo.optimizer import Trial
from olympus.studies.searchspace.main import create_valid_curves_xarray
from olympus.studies.variance.main import create_trials, fetch_vars_stats
from olympus.studies.hpo.main import (
    generate_grid_search, generate_noisy_grid_search, generate_random_search,
    generate_bayesopt, generate_hpos, register_hpo, register_hpos, env,
    fetch_hpos_valid_curves, fetch_all_hpo_stats)
from olympus.studies.simul.main import (
    reset_pool_size, randomize_seeds, IDENTITY_SIZE, remaining,
    generate_biased_replicates, convert_data_to_xy, cutoff, fit_model, convert_samples_to_x,
    simulate_hpo, generate_simulated_fix, generate_simulated_replicates, generate_replicates,
    get_ready_configs, register, fetch_hpos_replicates, consolidate_results, get_configs_to_replicate)
from olympus.observers.msgtracker import MSGQTracker, METRIC_QUEUE, METRIC_ITEM, metric_logger
from olympus.hpo.parallel import (
    exec_remote_call, make_remote_call, RESULT_QUEUE, WORK_QUEUE, WORK_ITEM, HPO_ITEM)
from olympus.hpo.worker import TrialWorker
from olympus.hpo import Fidelity

import pytest



URI = 'mongo://127.0.0.1:27017'
DATABASE = 'olympus'

NAMESPACE = 'test-hpo'
CONFIG = {
    'name': 'random_search',
    'seed': 1, 'count': 1,
    'fidelity': Fidelity(1, 10, name='epoch').to_dict(),
    'space': {
        'a': 'uniform(lower=-1, upper=1)',
        'b': 'uniform(lower=-1, upper=1)',
        'c': 'uniform(lower=-1, upper=1)',
        'epoch': 'uniform(lower=1, upper=10)'
    }
}
DEFAULTS = {}


def foo(uid, a, b, c, d, e=1, epoch=0, experiment_name=NAMESPACE, client=None):
    result = a + 2 * b - c ** 2 + d + e
    for i in range(epoch + 1):
        data = {'obj': i + result, 'valid': i + result, 'uid': uid, 'epoch': i}
        client.push(METRIC_QUEUE, experiment_name, data, mtype=METRIC_ITEM)
    # print(i, data, NAMESPACE, epoch + 1)
    return result + i


@pytest.fixture
def clean_mongodb():
    client = pymongo.MongoClient(URI.replace('mongo', 'mongodb'))
    client[DATABASE][WORK_QUEUE].drop()
    client[DATABASE][RESULT_QUEUE].drop()
    client[DATABASE][METRIC_QUEUE].drop()


@pytest.fixture
def client():
    return new_client(URI, DATABASE)


def generate_mocked_replicates(num_replicates, num_experiments=5):
    
    space = {
        'a': 'uniform(lower=-1, upper=1)',
        'b': 'uniform(lower=-1, upper=1)',
        'c': 'uniform(lower=-1, upper=1)'}
    variables = {'d': 3}
    defaults = {'d': 1, 'e': 2, 'epoch': 5}
    seed = 2
    hpo = 'random_search'
    objective = 'obj'
    fidelity = Fidelity(5, 5, name='epoch').to_dict()

    surrogate_budget = 10
    hpo_budget = 5
    
    configs = generate_hpos(
        list(range(num_experiments)), [hpo], budget=surrogate_budget,
        fidelity=fidelity, search_space=space, namespace=NAMESPACE, defaults=defaults)

    to_replicate = get_configs_to_replicate(configs, num_experiments)
    randomize_seeds(configs['random_search'], variables, seed)

    hpos_ready = dict(random_search=[])
    data = dict(random_search=dict())
    for hpo_namespace, config in configs['random_search'].items():
        hpos_ready['random_search'].append(hpo_namespace)
        data['random_search'][hpo_namespace] = build_data(
            surrogate_budget, variables, defaults, space)

    ready_configs = get_ready_configs(hpos_ready, configs, to_replicate)

    return generate_replicates(
        ready_configs, data, variables, objective, hpo_budget, num_replicates, early_stopping=False)


def build_data(budget, variables, defaults, space):

    epochs = 5
    # defaults = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    # params = {'c': 2, 'd': 3, 'epoch': epochs}
    n_vars = len(variables)

    objectives = numpy.arange(budget * (epochs + 1))
    numpy.random.RandomState(0).shuffle(objectives)
    objectives = objectives.reshape((epochs + 1, budget, 1))

    params = Space.from_dict(space).sample(budget, seed=1)

    trials = OrderedDict()
    for trial_params in params:
        config = copy.deepcopy(dict(list(variables.items()) + list(defaults.items())))
        config.update(trial_params)
        config['uid'] = compute_identity(config, IDENTITY_SIZE)
        # NOTE: We don't need objectives
        trials[config['uid']] = Trial(config)

    metrics = dict()
    for trial_i, trial_uid in enumerate(trials.keys()):
        metrics[trial_uid] = [
            {'epoch': i, 'obj': objectives[i, trial_i, 0]}
            for i in range(epochs + 1)]

    data = []
    param_names = list(sorted(space.keys()))

    return create_valid_curves_xarray(
        trials, metrics, sorted(variables.keys()), epochs, param_names, seed=1)


def test_reset_pool_size():
    space = {
        'a': 'uniform(lower=-1, upper=1)',
        'b': 'uniform(lower=-1, upper=1)',
        'c': 'uniform(lower=-1, upper=1)',
        'd': 'uniform(lower=-1, upper=1)'}
    defaults = {'d': 1, 'e': 2}
    num_experiments = 5
    hpo = 'random_search'
    fidelity = Fidelity(1, 1, name='epoch').to_dict()
    
    configs = generate_hpos(
        list(range(num_experiments)), [hpo], budget=200,
        fidelity=fidelity, search_space=space, namespace=NAMESPACE, defaults=defaults)
    reset_pool_size(configs['random_search'])

    for config in configs['random_search'].values():
        assert config['pool_size'] is None


def test_randomize_seeds():
    space = {
        'a': 'uniform(lower=-1, upper=1)',
        'b': 'uniform(lower=-1, upper=1)',
        'c': 'uniform(lower=-1, upper=1)'}
    variables = ['d', 'e']
    defaults = {}
    seed = 2
    num_experiments = 5
    hpo = 'random_search'
    fidelity = Fidelity(1, 1, name='epoch').to_dict()
    
    configs = generate_hpos(
        list(range(num_experiments)), [hpo], budget=200,
        fidelity=fidelity, search_space=space, namespace=NAMESPACE, defaults=defaults)

    randomize_seeds(configs['random_search'], variables, seed, compute_id=True)

    rng = numpy.random.RandomState(seed)
    for config in configs['random_search'].values():
        for variable in variables:
            assert config['defaults'][variable] == rng.randint(2**30)
        uid = config['defaults'].pop('uid')
        assert uid == compute_identity(config['defaults'], IDENTITY_SIZE)

    randomize_seeds(configs['random_search'], variables, seed, compute_id=False)
    rng = numpy.random.RandomState(seed)
    for config in configs['random_search'].values():
        for variable in variables:
            assert config['defaults'][variable] == rng.randint(2**30)
        assert 'uid' not in config['defaults']


def test_remaining():
    hpo_stats = {
        'random-search-s-1': {
            'read': 3,
            'actioned': 1,
            'error': 0, 
            'retry': 0,
            'count': 3},
        'random-search-s-2': {
            'read': 3,
            'actioned': 3,
            'error': 0, 
            'retry': 0,
            'count': 3},
        'random-search-s-3': {
            'read': 0,
            'actioned': 0,
            'error': 0, 
            'retry': 0,
            'count': 0}
        }

    assert remaining(hpo_stats)
    
    hpo_stats['random-search-s-1']['actioned'] = 3

    assert remaining(hpo_stats)

    hpo_stats['random-search-s-3']['count'] = 3

    assert remaining(hpo_stats)

    hpo_stats['random-search-s-3']['actioned'] = 3

    assert not remaining(hpo_stats)


def test_generate_biased_replicates_early_stopping():
    space = {
        'a': 'uniform(lower=-1, upper=1)',
        'b': 'uniform(lower=-1, upper=1)',
        'c': 'uniform(lower=-1, upper=1)'}
    variables = {'d': 1}
    defaults = {'e': 2}
    seed = 2
    num_experiments = 5
    hpo = 'random_search'
    objective = 'obj'
    num_replicates = 10
    fidelity = Fidelity(1, 1, name='epoch').to_dict()

    surrogate_budget = 10
    hpo_budget = 5
    
    configs = generate_hpos(
        list(range(num_experiments)), [hpo], budget=surrogate_budget,
        fidelity=fidelity, search_space=space, namespace=NAMESPACE, defaults=defaults)

    randomize_seeds(configs['random_search'], variables, seed)

    data = build_data(surrogate_budget, variables, defaults, space)

    replicates = generate_biased_replicates(
        data, configs['random_search'][f'{NAMESPACE}-random-search-s-0'], variables, objective,
        num_replicates, hpo_budget, early_stopping=True)

    best_trial_index = 6
    rng = numpy.random.RandomState(
        configs['random_search'][f'{NAMESPACE}-random-search-s-0']['seed'])
    for replicate in replicates:
        should_be = copy.deepcopy(defaults)
        for param in space.keys():
            assert replicate[param] == float(data.sel(order=best_trial_index)[param].values)
            should_be[param] = replicate[param]
        for variable in variables:
            assert replicate[variable] == rng.randint(2**30)
            should_be[variable] = replicate[variable]

        assert replicate['uid'] == compute_identity(should_be, IDENTITY_SIZE)


def test_generate_biased_replicates_last_epoch():
    space = {
        'a': 'uniform(lower=-1, upper=1)',
        'b': 'uniform(lower=-1, upper=1)',
        'c': 'uniform(lower=-1, upper=1)'}
    variables = {'d': 2, 'e': 1}
    defaults = {'d': 1, 'e': 2}
    seed = 2
    num_experiments = 5
    hpo = 'random_search'
    objective = 'obj'
    num_replicates = 10
    fidelity = Fidelity(1, 1, name='epoch').to_dict()

    surrogate_budget = 10
    hpo_budget = 5
    
    configs = generate_hpos(
        list(range(num_experiments)), [hpo], budget=surrogate_budget,
        fidelity=fidelity, search_space=space, namespace=NAMESPACE, defaults=defaults)

    randomize_seeds(configs['random_search'], variables, seed)

    data = build_data(surrogate_budget, variables, defaults, space)

    replicates = generate_biased_replicates(
        data, configs['random_search'][f'{NAMESPACE}-random-search-s-0'], variables, objective,
        num_replicates, hpo_budget, early_stopping=False)

    best_trial_index = 6
    rng = numpy.random.RandomState(
        configs['random_search'][f'{NAMESPACE}-random-search-s-0']['seed'])
    for replicate in replicates:
        should_be = copy.deepcopy(defaults)
        for param in space.keys():
            assert replicate[param] == float(data.sel(order=best_trial_index)[param].values)
            should_be[param] = replicate[param]
        for variable in variables:
            assert replicate[variable] == rng.randint(2**30)
            should_be[variable] = replicate[variable]

        assert replicate['uid'] == compute_identity(should_be, IDENTITY_SIZE)


def test_convert_data_to_xy():
    space = {
        'a': 'uniform(lower=-1, upper=1)',
        'b': 'uniform(lower=-1, upper=1)',
        'c': 'loguniform(lower=1, upper=10)'}
    variables = {'d': 2, 'e': 1}
    defaults = {'d': 1, 'e': 2}
    seed = 2
    num_experiments = 5
    hpo = 'random_search'
    objective = 'obj'
    num_replicates = 10
    fidelity = Fidelity(1, 1, name='epoch').to_dict()

    surrogate_budget = 10
    hpo_budget = 5
    
    configs = generate_hpos(
        list(range(num_experiments)), [hpo], budget=surrogate_budget,
        fidelity=fidelity, search_space=space, namespace=NAMESPACE, defaults=defaults)

    randomize_seeds(configs['random_search'], variables, seed)

    data = build_data(surrogate_budget, variables, defaults, space)

    X, y = convert_data_to_xy(data, space, objective, early_stopping=False)

    assert numpy.array_equal(X[:, 0], data['a'].values.reshape(-1))
    assert numpy.array_equal(X[:, 1], data['b'].values.reshape(-1))
    assert numpy.array_equal(X[:, 2], numpy.log(data['c'].values.reshape(-1)))
    assert numpy.array_equal(y, data[objective].isel(epoch=-1).values.reshape(-1))
    assert y.shape == (surrogate_budget, )


def test_cutoff():
    N = 10
    D = 3
    X = numpy.random.normal(size=(N, D))
    y = numpy.arange(10)
    numpy.random.RandomState(0).shuffle(y)
    cX, cy = cutoff(X, y, percentile=0.8)
    assert max(cy) == 7
    assert len(cy) == 8
    assert numpy.array_equal(X[numpy.argsort(y)[:8], :], cX)


def test_fit_model():
    space = {
        'a': 'uniform(lower=-1, upper=1)',
        'b': 'uniform(lower=-1, upper=1)',
        'c': 'loguniform(lower=1, upper=10)'}
    variables = {'d': 2, 'e': 1}
    defaults = {'d': 1, 'e': 2}
    seed = 2
    num_experiments = 5
    hpo = 'random_search'
    objective = 'obj'
    num_replicates = 10
    fidelity = Fidelity(1, 1, name='epoch').to_dict()

    surrogate_budget = 10
    hpo_budget = 5
    
    configs = generate_hpos(
        list(range(num_experiments)), [hpo], budget=surrogate_budget,
        fidelity=fidelity, search_space=space, namespace=NAMESPACE, defaults=defaults)

    randomize_seeds(configs['random_search'], variables, seed)

    data = build_data(surrogate_budget, variables, defaults, space)

    X, y = convert_data_to_xy(data, space, objective)

    fit_model(X, y, space, seed=1)


def test_convert_samples_to_x():
    space = {
        'a': 'uniform(lower=-1, upper=1)',
        'b': 'uniform(lower=-1, upper=1)',
        'c': 'loguniform(lower=1, upper=10)'}

    samples = Space.from_dict(space).sample(10, seed=1)
    X = convert_samples_to_x(samples, space)

    assert X.shape == (10, 3)
    assert X[0, 0] == samples[0]['a']
    assert X[0, 1] == samples[0]['b']
    assert X[0, 2] == numpy.log(samples[0]['c'])


def test_simulate_hpo():
    # fit a model
    # simulate
    # test what? ...
    space = {
        'a': 'uniform(lower=-1, upper=1)',
        'b': 'uniform(lower=-1, upper=1)',
        'c': 'loguniform(lower=1, upper=10)'}
    variables = {'d': 2, 'e': 1}
    defaults = {'d': 1, 'e': 2}
    seed = 2
    num_experiments = 5
    hpo = 'random_search'
    objective = 'obj'
    num_replicates = 10
    fidelity = Fidelity(1, 1, name='epoch').to_dict()

    surrogate_budget = 10
    hpo_budget = 5
    
    configs = generate_hpos(
        list(range(num_experiments)), [hpo], budget=surrogate_budget,
        fidelity=fidelity, search_space=space, namespace=NAMESPACE, defaults=defaults)

    randomize_seeds(configs['random_search'], variables, seed)

    data = build_data(surrogate_budget, variables, defaults, space)

    X, y = convert_data_to_xy(data, space, objective)

    model = fit_model(X, y, space, seed=1)

    sample = simulate_hpo(model, space, hpo_budget, 
                          configs['random_search'][f'{NAMESPACE}-random-search-s-0']['seed'])
    assert sample.keys() == space.keys()


def test_generate_simulated_fix():
    space = {
        'a': 'uniform(lower=-1, upper=1)',
        'b': 'uniform(lower=-1, upper=1)',
        'c': 'loguniform(lower=1, upper=10)'}
    variables = {'d': 2, 'e': 1}
    defaults = {'d': 1, 'e': 2}
    seed = 2
    num_experiments = 5
    hpo = 'random_search'
    objective = 'obj'
    num_replicates = 10
    fidelity = Fidelity(1, 1, name='epoch').to_dict()

    surrogate_budget = 10
    hpo_budget = 5
    
    configs = generate_hpos(
        list(range(num_experiments)), [hpo], budget=surrogate_budget,
        fidelity=fidelity, search_space=space, namespace=NAMESPACE, defaults=defaults)

    randomize_seeds(configs['random_search'], variables, seed)

    data = build_data(surrogate_budget, variables, defaults, space)

    config = configs['random_search'][f'{NAMESPACE}-random-search-s-0']
    # Make sure the defaults have been replaced by randomized seeds
    assert config['defaults']['d'] != defaults['d']
    assert config['defaults']['e'] != defaults['e']

    replicates = generate_simulated_fix(
        data, config, variables, objective, hpo_budget,
        num_replicates, early_stopping=False)

    assert len(replicates) == num_replicates
    for i in range(1, num_replicates):
        assert replicates[i]['a'] != replicates[0]['a']
        assert replicates[i]['b'] != replicates[0]['b']
        assert replicates[i]['c'] != replicates[0]['c']
        assert replicates[i]['uid'] != replicates[0]['uid']
        assert replicates[i]['d'] == config['defaults']['d']
        assert replicates[i]['e'] == config['defaults']['e']


def test_generate_simulated_replicates():
    num_replicates = 10
    fake_simulated_replicates = []
    variables = ['d', 'e']
    for i in range(num_replicates):
        replicate = {'a': i, 'b': i, 'c': i, 'd': 1, 'e': 1}
        replicate['uid'] = compute_identity(replicate, IDENTITY_SIZE)
        fake_simulated_replicates.append(replicate)

    replicates = generate_simulated_replicates(fake_simulated_replicates, {'seed': 1}, variables)

    assert len(replicates) == num_replicates
    for fix_replicate, var_replicate in zip(fake_simulated_replicates, replicates):
        assert fix_replicate['a'] == var_replicate['a']
        assert fix_replicate['b'] == var_replicate['b']
        assert fix_replicate['c'] == var_replicate['c']
        assert fix_replicate['d'] != var_replicate['d']
        assert fix_replicate['e'] != var_replicate['e']


def test_generate_replicates():

    num_replicates = 10
    num_experiments = 5
    replicates = generate_mocked_replicates(num_replicates, num_experiments)

    assert len(replicates) == 1

    replicates = replicates['random_search']
    assert len(replicates) == num_experiments
    print(replicates.keys())
    assert len(replicates[f'{NAMESPACE}-random-search-s-0']) == 3
    assert len(replicates[f'{NAMESPACE}-random-search-s-1']) == 3
    assert len(replicates[f'{NAMESPACE}-random-search-s-0']['biased']) == num_replicates
    assert len(replicates[f'{NAMESPACE}-random-search-s-0']['simul-fix']) == num_replicates
    assert len(replicates[f'{NAMESPACE}-random-search-s-0']['simul-free']) == num_replicates
    assert len(replicates[f'{NAMESPACE}-random-search-s-1']['biased']) == num_replicates
    assert len(replicates[f'{NAMESPACE}-random-search-s-1']['simul-fix']) == num_replicates
    assert len(replicates[f'{NAMESPACE}-random-search-s-1']['simul-free']) == num_replicates

    # Make sur defaults are passed properly to replicates
    for simul_type in ['biased', 'simul-free', 'simul-fix']:
        for i in range(num_replicates):
            assert replicates[f'{NAMESPACE}-random-search-s-0'][simul_type][i]['d'] != 1
            assert replicates[f'{NAMESPACE}-random-search-s-0'][simul_type][i]['e'] == 2


@pytest.mark.usefixtures('clean_mongodb')
def test_register_uniques(client):
    num_replicates = 10
    num_experiments = 5
    replicates = generate_mocked_replicates(num_replicates, num_experiments)

    def count_registered():
        status = fetch_vars_stats(client, NAMESPACE)
        return sum(status[key]['count'] for key in status.keys())

    assert count_registered() == 0
    register(client, foo, NAMESPACE, replicates)
    assert count_registered() == num_experiments * num_replicates * 3  # (biased+simfree+simfixed)


@pytest.mark.usefixtures('clean_mongodb')
def test_register_resume(client):
    num_replicates = 10
    num_experiments = 5
    replicates = generate_mocked_replicates(num_replicates, num_experiments)

    def count_registered():
        status = fetch_vars_stats(client, NAMESPACE)
        return sum(status[key]['count'] for key in status.keys())

    assert count_registered() == 0
    register(client, foo, NAMESPACE, replicates)
    assert count_registered() == num_experiments * num_replicates * 3  # (biased+simfree+simfixed)

    # Resume with 10 more replicates per configs this time.
    replicates = generate_mocked_replicates(num_replicates + 10, num_experiments)
    register(client, foo, NAMESPACE, replicates)
    assert count_registered() == num_experiments * (num_replicates + 10) * 3


@pytest.mark.usefixtures('clean_mongodb')
def test_fetch_hpos_replicates(client):

    num_experiments = 5
    num_replicates = 10
    space = {
        'a': 'uniform(lower=-1, upper=1)',
        'b': 'uniform(lower=-1, upper=1)',
        'c': 'uniform(lower=-1, upper=1)'}
    variables = {'d': 5}
    defaults = {'e': 2, 'epoch': 5}
    seed = 2
    hpo = 'random_search'
    objective = 'obj'
    fidelity = Fidelity(5, 5, name='epoch').to_dict()

    surrogate_budget = 10
    hpo_budget = 5
    
    configs = generate_hpos(
        list(range(num_experiments)), [hpo], budget=surrogate_budget,
        fidelity=fidelity, search_space=space, namespace=NAMESPACE, defaults=defaults)

    to_replicate = get_configs_to_replicate(configs, num_experiments)

    reset_pool_size(configs['random_search'])
    randomize_seeds(configs['random_search'], variables, seed)

    variable_names = list(sorted(variables.keys()))

    hpo_stats = fetch_all_hpo_stats(client, NAMESPACE)

    namespaces = register_hpos(
        client, NAMESPACE, foo, configs, defaults, hpo_stats)

    worker = TrialWorker(URI, DATABASE, 0, None)
    worker.max_retry = 0
    worker.timeout = 0.02
    worker.run()

    data = defaultdict(dict)
    hpos_ready, remainings = fetch_hpos_valid_curves(client, namespaces, variable_names, data)

    ready_configs = get_ready_configs(hpos_ready, configs, to_replicate)

    replicates = generate_replicates(
        ready_configs, data, variables, objective, hpo_budget, num_replicates, early_stopping=False)
    register(client, foo, NAMESPACE, replicates)

    worker = TrialWorker(URI, DATABASE, 0, None)
    worker.max_retry = 0
    worker.timeout = 0.02
    worker.run()
    print(fetch_vars_stats(client, NAMESPACE))

    data = fetch_hpos_replicates(client, configs, replicates, variable_names, space, data)

    assert len(data) == 1
    assert len(data['random_search']) == num_experiments
    assert len(data['random_search']['test-hpo-random-search-s-0']) == 4

    hpo_reps = data['random_search']['test-hpo-random-search-s-0']
    assert hpo_reps['ideal'].obj.shape == (6, surrogate_budget, 1)
    assert hpo_reps['biased'].obj.shape == (6, num_replicates, 1)
    assert hpo_reps['simul-fix'].obj.shape == (6, num_replicates, 1)
    assert hpo_reps['simul-free'].obj.shape == (6, num_replicates, 1)

    print(hpo_reps['ideal']['d'].values.shape)
    print(hpo_reps['ideal']['d'].values)

    def count_unique(attr):
        return len(set(attr.values.reshape(-1).tolist()))

    # Test sources of variation
    # NOTE: In ideal, source of variation will vary across ideal after consolidation 
    #       but it stays fixed during the HPO itself
    assert count_unique(hpo_reps['ideal']['d']) == 1
    assert count_unique(hpo_reps['biased']['d']) == num_replicates
    assert count_unique(hpo_reps['simul-free']['d']) == num_replicates
    assert count_unique(hpo_reps['simul-fix']['d']) == 1

    # Test HPs
    assert count_unique(hpo_reps['ideal']['a']) == num_replicates
    assert count_unique(hpo_reps['biased']['a']) == 1
    assert count_unique(hpo_reps['simul-free']['a']) == num_replicates
    assert count_unique(hpo_reps['simul-fix']['a']) == num_replicates
    assert numpy.allclose(hpo_reps['simul-free']['a'].values, hpo_reps['simul-fix']['a'].values)


@pytest.mark.usefixtures('clean_mongodb')
def test_consolidate_results(client):
    num_experiments = 5
    num_replicates = 10
    space = {
        'a': 'uniform(lower=-1, upper=1)',
        'b': 'uniform(lower=-1, upper=1)',
        'c': 'uniform(lower=-1, upper=1)'}
    variables = {'d': 5}
    defaults = {'e': 2, 'epoch': 5}
    seed = 2
    hpo = 'random_search'
    objective = 'obj'
    fidelity = Fidelity(5, 5, name='epoch').to_dict()

    surrogate_budget = 10
    hpo_budget = 5
    
    configs = generate_hpos(
        list(range(num_experiments)), [hpo], budget=surrogate_budget,
        fidelity=fidelity, search_space=space, namespace=NAMESPACE, defaults=defaults)

    to_replicate = get_configs_to_replicate(configs, num_experiments)

    reset_pool_size(configs['random_search'])
    randomize_seeds(configs['random_search'], variables, seed)

    variable_names = list(sorted(variables.keys()))

    hpo_stats = fetch_all_hpo_stats(client, NAMESPACE)

    namespaces = register_hpos(
        client, NAMESPACE, foo, configs, defaults, hpo_stats)

    worker = TrialWorker(URI, DATABASE, 0, None)
    worker.max_retry = 0
    worker.timeout = 0.02
    worker.run()

    data = defaultdict(dict)
    hpos_ready, remainings = fetch_hpos_valid_curves(client, namespaces, variable_names, data)

    ready_configs = get_ready_configs(hpos_ready, configs, to_replicate)

    replicates = generate_replicates(
        ready_configs, data, variables, objective, hpo_budget, num_replicates, early_stopping=False)
    register(client, foo, NAMESPACE, replicates)

    worker = TrialWorker(URI, DATABASE, 0, None)
    worker.max_retry = 0
    worker.timeout = 0.02
    worker.run()
    print(fetch_vars_stats(client, NAMESPACE))

    data = fetch_hpos_replicates(client, configs, replicates, variable_names, space, data)
    data = consolidate_results(data)

    assert len(data) == 1
    assert len(data['random_search']) == 4

    hpo_reps = data['random_search']
    assert hpo_reps['ideal'].obj.shape == (6, surrogate_budget, num_experiments)
    assert hpo_reps['biased'].obj.shape == (6, num_replicates, num_experiments)
    assert hpo_reps['simul-fix'].obj.shape == (6, num_replicates, num_experiments)
    assert hpo_reps['simul-free'].obj.shape == (6, num_replicates, num_experiments)

    def count_unique(attr):
        return len(set(attr.values.reshape(-1).tolist()))

    # Test sources of variation
    # NOTE: In ideal, source of variation will vary across ideal after consolidation 
    #       but it stays fixed during the HPO itself
    assert count_unique(hpo_reps['ideal']['d']) == num_experiments
    assert count_unique(hpo_reps['biased']['d']) == (num_replicates * num_experiments)
    assert count_unique(hpo_reps['simul-free']['d']) == (num_replicates * num_experiments)
    assert count_unique(hpo_reps['simul-fix']['d']) == num_experiments

    # Test HPs
    assert count_unique(hpo_reps['ideal']['a']) == (num_experiments * surrogate_budget)
    assert count_unique(hpo_reps['biased']['a']) == num_experiments
    assert count_unique(hpo_reps['simul-free']['a']) == (num_replicates * num_experiments)
    assert count_unique(hpo_reps['simul-fix']['a']) == (num_replicates * num_experiments)
    assert numpy.allclose(hpo_reps['simul-free']['a'].values, hpo_reps['simul-fix']['a'].values)
