import copy
from collections import defaultdict

import numpy
import xarray

import pymongo

from sspace.space import compute_identity

from msgqueue.backends import new_client

from studies.hpo.main import (
    generate_grid_search, generate_noisy_grid_search, generate_random_search,
    generate_bayesopt, generate_hpos, register_hpo, register_hpos, fetch_hpos_valid_curves, consolidate_results, save_results, load_results)
from olympus.observers.msgtracker import METRIC_QUEUE, METRIC_ITEM
from olympus.hpo.parallel import (
    RESULT_QUEUE, WORK_QUEUE, HPO_ITEM)
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
        'd': 'uniform(lower=-1, upper=1)',
        'epoch': 'uniform(lower=1, upper=10)'
    }
}
DEFAULTS = {}


def foo(uid, a, b, c, d, e=1, epoch=0, experiment_name=NAMESPACE, client=None):
    result = a + 2 * b - c ** 2 + d + e
    for i in range(epoch + 1):
        data = {'obj': i + result, 'valid': i + result, 'uid': uid, 'epoch': i}
        client.push(METRIC_QUEUE, experiment_name, data, mtype=METRIC_ITEM)
    print(i, data, NAMESPACE, epoch + 1)
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


def build_data(size):
    variables = 'abc'
    order = range(size)
    n_vars = len(variables)
    n_seeds = len(order)
    objectives = numpy.arange(n_vars * n_seeds)
    numpy.random.RandomState(0).shuffle(objectives)
    objectives = objectives.reshape((n_vars, n_seeds))
    uids = numpy.zeros((n_vars, n_seeds)).astype(str)
    seeds = numpy.arange(n_vars * n_seeds * n_vars).reshape((n_vars, n_seeds, n_vars)).astype(int)

    coords = {
        'vars': list(variables),
        'order': order,
        'uid': (('vars', 'order'), uids)}

    for i, variable in enumerate(variables):
        coords[variable] = (('vars', 'order'), seeds[:, :, i])

    objectives = xarray.DataArray(
        objectives,
        dims=['vars', 'order'],
        coords=coords)

    data_vars = {'objectives': (('vars', 'order'), objectives)}

    data = xarray.Dataset(
        data_vars=data_vars,
        coords=coords)

    return data


def test_generate_grid_search():
    defaults = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    budget = 200
    fidelity = 'fidelity(1, 10)'
    num_experiments = 10
    search_space = {
        'a': 'uniform(-1, 1)',
        'b': 'uniform(-1, 1)',
        'c': 'uniform(-1, 1)',
        'd': 'uniform(-1, 1)'
    }
    configs = generate_grid_search(budget, fidelity, search_space, range(num_experiments))

    assert len(configs) == 1

    assert configs[0]['name'] == 'grid_search'
    assert configs[0]['n_points'] == 4
    assert configs[0]['space'] == search_space
    assert configs[0]['seed'] == 1
    assert configs[0]['namespace'] == 'grid-search-p-4'
    assert configs[0].pop('uid') == compute_identity(configs[0], 16)


def test_generate_grid_search_n_points():
    defaults = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    fidelity = 'fidelity(1, 10)'
    num_experiments = 10
    search_space = {
        'a': 'uniform(-1, 1)',
        'b': 'uniform(-1, 1)',
        'lr': 'uniform(-1, 1)'
    }
    
    # Handle budgets that does not match grid size
    configs = generate_grid_search(4 ** 3 - 1, fidelity, search_space, range(num_experiments))
    assert configs[0]['n_points'] == 4
    configs = generate_grid_search(4 ** 3, fidelity, search_space, range(num_experiments))
    assert configs[0]['n_points'] == 4
    configs = generate_grid_search(4 ** 3 + 1, fidelity, search_space, range(num_experiments))
    assert configs[0]['n_points'] == 5

    # Handle larger search space
    search_space['c'] = 'uniform(-1, 1)'
    configs = generate_grid_search(4 ** 4 - 1, fidelity, search_space, range(num_experiments))
    assert configs[0]['n_points'] == 4
    configs = generate_grid_search(4 ** 4, fidelity, search_space, range(num_experiments))
    assert configs[0]['n_points'] == 4
    configs = generate_grid_search(4 ** 4 + 1, fidelity, search_space, range(num_experiments))
    assert configs[0]['n_points'] == 5


def test_generate_noisy_grid_search():
    defaults = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    budget = 200
    fidelity = 'fidelity(1, 10)'
    num_experiments = 10
    search_space = {
        'a': 'uniform(-1, 1)',
        'b': 'uniform(-1, 1)',
        'c': 'uniform(-1, 1)',
        'lr': 'uniform(-1, 1)'
    }
    configs = generate_noisy_grid_search(budget, fidelity, search_space, range(num_experiments))

    assert len(configs) == num_experiments

    for i in range(num_experiments):
        assert configs[i]['name'] == 'noisy_grid_search'
        assert configs[i]['n_points'] == 4
        assert configs[i]['space'] == search_space
        assert configs[i]['seed'] == i
        assert configs[i]['namespace'] == f'noisy-grid-search-p-4-s-{i}'
        assert configs[i].pop('uid') == compute_identity(configs[i], 16)


def test_generate_random_search():
    defaults = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    budget = 200
    fidelity = 'fidelity(1, 10)'
    num_experiments = 10
    search_space = {
        'a': 'uniform(-1, 1)',
        'b': 'uniform(-1, 1)',
        'c': 'uniform(-1, 1)',
        'lr': 'uniform(-1, 1)'
    }
    configs = generate_random_search(budget, fidelity, search_space, range(num_experiments))

    assert len(configs) == num_experiments

    for i in range(num_experiments):
        assert configs[i]['name'] == 'random_search'
        assert configs[i]['space'] == search_space
        assert configs[i]['seed'] == i
        assert configs[i]['namespace'] == f'random-search-s-{i}'
        assert configs[i].pop('uid') == compute_identity(configs[i], 16)


@pytest.mark.skip(reason='Requires adaptive budget')
def test_generate_hyperband():
    assert False


def test_generate_bayesopt():
    defaults = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    budget = 200
    fidelity = 'fidelity(1, 10)'
    num_experiments = 10
    search_space = {
        'a': 'uniform(-1, 1)',
        'b': 'uniform(-1, 1)',
        'c': 'uniform(-1, 1)',
        'lr': 'uniform(-1, 1)'
    }
    configs = generate_bayesopt(budget, fidelity, search_space, range(num_experiments))

    assert len(configs) == num_experiments

    for i in range(num_experiments):
        rng = numpy.random.RandomState(i)
        assert configs[i]['name'] == 'robo'
        assert configs[i]['space'] == search_space
        assert configs[i]['n_init'] == 20
        assert configs[i]['count'] == budget
        assert configs[i]['model_seed'] == rng.randint(2**30)
        assert configs[i]['prior_seed'] == rng.randint(2**30)
        assert configs[i]['init_seed'] == rng.randint(2**30)
        assert configs[i]['maximizer_seed'] == rng.randint(2**30)
        assert configs[i]['namespace'] == f'bayesopt-s-{i}'
        assert configs[i].pop('uid') == compute_identity(configs[i], 16)


def test_generate_hpos():
    hpos = ['grid_search', 'nudged_grid_search', 'noisy_grid_search', 'random_search',
            'bayesopt']
    #       , 'hyperband', 'bayesopt']
    num_experiments = 10
    budget = 200
    fidelity = 'fidelity(1, 10)'
    num_experiments = 10
    search_space = {
        'a': 'uniform(-1, 1)',
        'b': 'uniform(-1, 1)',
        'c': 'uniform(-1, 1)',
        'lr': 'uniform(-1, 1)'
    }
    defaults = {'d': 1, 'e': 2}

    configs = generate_hpos(
        range(num_experiments), hpos, budget, fidelity, search_space, NAMESPACE, defaults)

    assert len(configs) == len(hpos)
    assert len(configs['grid_search']) == 1
    assert len(configs['nudged_grid_search']) == 1
    assert configs['grid_search'][f'{NAMESPACE}-grid-search-p-4']['name'] == 'grid_search'
    assert configs['grid_search'][f'{NAMESPACE}-grid-search-p-4'].get('nudge') is None
    assert (configs['nudged_grid_search'][f'{NAMESPACE}-grid-search-nudged-p-4']['name'] ==
            'grid_search')
    assert configs['nudged_grid_search'][f'{NAMESPACE}-grid-search-nudged-p-4']['nudge'] == 0.5
    for hpo in hpos[2:]:
        for i in range(num_experiments):
            if hpo == 'noisy_grid_search':
                namespace = '{}-noisy-grid-search-p-4-s-{}'.format(NAMESPACE, i)
            else:
                namespace = '{}-{}-s-{}'.format(NAMESPACE, hpo.replace('_', '-'), i)
            if hpo == 'bayesopt':
                rng = numpy.random.RandomState(i)
                assert configs[hpo][namespace]['name'] == 'robo'
                assert configs[hpo][namespace]['model_seed'] == rng.randint(2**30)
                assert configs[hpo][namespace]['prior_seed'] == rng.randint(2**30)
                assert configs[hpo][namespace]['init_seed'] == rng.randint(2**30)
                assert configs[hpo][namespace]['maximizer_seed'] == rng.randint(2**30)
            else:
                assert configs[hpo][namespace]['name'] == hpo
                assert configs[hpo][namespace]['seed'] == i


@pytest.mark.usefixtures('clean_mongodb')
def test_register_hpos(client):
    namespace = 'test-hpo'
    hpos = ['grid_search', 'nudged_grid_search', 'noisy_grid_search', 'random_search',
            'bayesopt']
    #       , 'hyperband', 'bayesopt']
    num_experiments = 10
    budget = 200
    fidelity = Fidelity(1, 10, name='d').to_dict()
    num_experiments = 2
    search_space = {
        'a': 'uniform(-1, 1)',
        'b': 'uniform(-1, 1)',
        'c': 'uniform(-1, 1)',
        'd': 'uniform(-1, 1)'
    }
    defaults = {'e': 2}

    configs = generate_hpos(
        range(num_experiments), hpos, budget, fidelity, search_space, NAMESPACE, defaults)

    stats = {}

    assert client.monitor().read_count(WORK_QUEUE, namespace, mtype=HPO_ITEM) == 0
    new_namespaces = register_hpos(client, namespace, foo, configs, defaults, stats)
    assert len(set(new_namespaces)) == len(configs)
    for hpo, hpo_namespaces in new_namespaces.items():
        for i, hpo_namespace in enumerate(hpo_namespaces):
            messages = client.monitor().messages(WORK_QUEUE, hpo_namespace, mtype=HPO_ITEM)
            assert len(messages) == 1
            assert messages[0].message['hpo']['kwargs'] == configs[hpo][hpo_namespace]
            assert messages[0].message['work']['kwargs'] == defaults


@pytest.mark.usefixtures('clean_mongodb')
def test_register_hpos_resume(client, monkeypatch):
    namespace = 'test-hpo'
    hpos = ['grid_search', 'nudged_grid_search', 'noisy_grid_search', 'random_search',
            'bayesopt']
    #       , 'hyperband', 'bayesopt']
    num_experiments = 10
    budget = 200
    fidelity = Fidelity(1, 10, name='d').to_dict()
    num_experiments = 2
    search_space = {
        'a': 'uniform(-1, 1)',
        'b': 'uniform(-1, 1)',
        'c': 'uniform(-1, 1)',
        'd': 'uniform(-1, 1)'
    }
    defaults = {}
    stats = {}

    configs = generate_hpos(
        range(num_experiments), hpos, budget, fidelity, search_space, NAMESPACE, defaults)

    assert client.monitor().read_count(WORK_QUEUE, namespace, mtype=HPO_ITEM) == 0
    new_namespaces = register_hpos(client, namespace, foo, configs, defaults, stats)
    assert len(set(new_namespaces)) == len(configs)

    print(new_namespaces)

    stats = {
        namespace: {}
        for namespace in sum(new_namespaces.values(), [])}

    more_configs = generate_hpos(range(num_experiments + 2), hpos, budget, fidelity, search_space,
                                 NAMESPACE, defaults)
            

    # Save new namespaces for test
    new_namespaces = defaultdict(list)
    def mock_register_hpo(client, namespace, function, config, defaults):
        new_namespaces[config['name']].append(namespace)
        return register_hpo(client, namespace, function, config, defaults)

    def flatten_configs(confs):
        return sum((list(configs.keys()) for configs in confs.values()), [])

    monkeypatch.setattr('olympus.studies.hpo.main.register_hpo', mock_register_hpo)
    namespaces = register_hpos(client, namespace, foo, more_configs, defaults, stats)
    assert (len(set(sum(new_namespaces.values(), []))) == 
            len(flatten_configs(more_configs)) - len(flatten_configs(configs)))

    # Verify new registered configs
    for hpo, configs in more_configs.items():
        for hpo_namespace, config in configs.items():
            messages = client.monitor().messages(WORK_QUEUE, hpo_namespace,
                                                 mtype=HPO_ITEM)
            assert len(messages) == 1
            assert messages[0].message['hpo']['kwargs'] == config


@pytest.mark.usefixtures('clean_mongodb')
def test_fetch_hpos_valid_results_first_time(client):
    config = copy.deepcopy(CONFIG)
    num_trials = 5
    config['count'] = num_trials
    config['fidelity'] = Fidelity(1, 1, name='epoch').to_dict()

    register_hpo(client, NAMESPACE + '1', foo, config, {'e': 2})
    register_hpo(client, NAMESPACE + '2', foo, config, {'e': 2})

    worker = TrialWorker(URI, DATABASE, 0, None)
    worker.max_retry = 0
    worker.timeout = 1
    worker.run()

    namespaces = {'hpo' + str(i): [NAMESPACE + str(i)] for i in range(1, 3)}

    data = defaultdict(dict)
    _ = fetch_hpos_valid_curves(client, namespaces, ['e'], data)

    assert len(data) == 2
    assert len(data['hpo1']) == 1
    assert len(data['hpo2']) == 1

    namespace = f'{NAMESPACE}1'
    assert data['hpo1'][namespace].attrs['namespace'] == namespace 
    assert data['hpo1'][namespace].epoch.values.tolist() == [0, 1]
    assert data['hpo1'][namespace].order.values.tolist() == list(range(num_trials))
    assert data['hpo1'][namespace].seed.values.tolist() == [1]
    assert data['hpo1'][namespace].params.values.tolist() == list('abcd')
    assert data['hpo1'][namespace].noise.values.tolist() == ['e']
    assert data['hpo1'][namespace].obj.shape == (2, num_trials, 1)
    assert data['hpo1'][namespace].valid.shape == (2, num_trials, 1)

    assert data['hpo1'][namespace] == data['hpo2'][f'{NAMESPACE}2']


@pytest.mark.usefixtures('clean_mongodb')
def test_fetch_hpos_valid_results_update(client):
    config = copy.deepcopy(CONFIG)
    num_trials = 5
    config['count'] = num_trials
    config['fidelity'] = Fidelity(0, 0, name='epoch').to_dict()

    namespaces = {f'hpo-{i}': [f'{NAMESPACE}-{i}-{j}' for j in range(1, 3)]
                  for i in range(1, 3)}

    def run_hpos(namespaces):
        for namespace in namespaces:
            register_hpo(client, namespace, foo, config, {'e': 2})

        worker = TrialWorker(URI, DATABASE, 0, None)
        worker.max_retry = 0
        worker.timeout = 1
        worker.run()

    run_hpos([namespaces['hpo-1'][0]])

    data = defaultdict(dict)
    hpos_ready, remainings = fetch_hpos_valid_curves(client, namespaces, ['e'], data)
    assert len(remainings) == 2
    assert len(remainings['hpo-1']) == 1
    assert len(remainings['hpo-2']) == 2

    assert len(hpos_ready) == 1
    assert len(hpos_ready['hpo-1']) == 1
    assert hpos_ready['hpo-1'][0] == f'{NAMESPACE}-1-1'

    assert len(data) == 1
    assert len(data['hpo-1']) == 1

    assert data['hpo-1'][f'{NAMESPACE}-1-1'].attrs['namespace'] == f'{NAMESPACE}-1-1'

    run_hpos([namespaces['hpo-1'][1], namespaces['hpo-2'][0]])

    hpos_ready, remainings = fetch_hpos_valid_curves(client, remainings, ['e'], data)
    assert len(remainings) == 1
    assert len(remainings['hpo-2']) == 1

    assert len(hpos_ready) == 2
    assert len(hpos_ready['hpo-1']) == 1
    assert hpos_ready['hpo-1'][0] == f'{NAMESPACE}-1-2'
    assert len(hpos_ready['hpo-2']) == 1
    assert hpos_ready['hpo-2'][0] == f'{NAMESPACE}-2-1'

    assert len(data) == 2
    assert len(data['hpo-1']) == 2
    assert len(data['hpo-2']) == 1

    assert data['hpo-1'][f'{NAMESPACE}-1-2'].attrs['namespace'] == f'{NAMESPACE}-1-2'
    assert data['hpo-2'][f'{NAMESPACE}-2-1'].attrs['namespace'] == f'{NAMESPACE}-2-1'

    run_hpos([namespaces['hpo-2'][1]])

    hpos_ready, remainings = fetch_hpos_valid_curves(client, remainings, ['e'], data)
    assert len(remainings) == 0
    assert len(hpos_ready) == 1

    assert len(hpos_ready['hpo-2']) == 1
    assert hpos_ready['hpo-2'][0] == f'{NAMESPACE}-2-2'

    assert len(data) == 2
    assert len(data['hpo-1']) == 2
    assert len(data['hpo-2']) == 2

    assert data['hpo-2'][f'{NAMESPACE}-2-2'].attrs['namespace'] == f'{NAMESPACE}-2-2'


@pytest.mark.usefixtures('clean_mongodb')
def test_consolidate_results(client):
    config = copy.deepcopy(CONFIG)
    num_trials = 5
    config['count'] = num_trials

    namespaces = {f'hpo-{i}': [f'{NAMESPACE}-{i}-{j}' for j in range(1, 3)]
                  for i in range(1, 3)}

    def run_hpos(namespaces):
        for i, namespace in enumerate(namespaces):
            config['seed'] = i
            register_hpo(client, namespace, foo, config, {'e': 2})

        worker = TrialWorker(URI, DATABASE, 0, None)
        worker.max_retry = 0
        worker.timeout = 1
        worker.run()

    run_hpos(sum(namespaces.values(), []))

    data = defaultdict(dict)
    hpos_ready, remainings = fetch_hpos_valid_curves(client, namespaces, ['e'], data)
    assert len(remainings) == 0
    assert len(hpos_ready) == 2

    data = consolidate_results(data)

    num_seeds = 2
    num_epochs = 10
    assert len(data) == 2
    assert data['hpo-1'].namespace.values.tolist() == [f'{NAMESPACE}-1-{i}' for i in range(1, 3)]
    assert data['hpo-1'].epoch.values.tolist() == list(range(num_epochs + 1))
    assert data['hpo-1'].order.values.tolist() == list(range(num_trials))
    assert data['hpo-1'].seed.values.tolist() == list(range(num_seeds))
    assert data['hpo-1'].params.values.tolist() == list('abcd')
    assert data['hpo-1'].noise.values.tolist() == ['e']
    assert data['hpo-1'].obj.shape == (num_epochs + 1, num_trials, num_seeds)
    assert data['hpo-1'].valid.shape == (num_epochs + 1, num_trials, num_seeds)


@pytest.mark.usefixtures('clean_mongodb')
def test_save_results(client):
    config = copy.deepcopy(CONFIG)
    num_trials = 5
    config['count'] = num_trials

    namespaces = {f'hpo-{i}': [f'{NAMESPACE}-{i}-{j}' for j in range(1, 3)]
                  for i in range(1, 3)}

    def run_hpos(namespaces):
        for i, namespace in enumerate(namespaces):
            config['seed'] = i
            register_hpo(client, namespace, foo, config, {'e': 2})

        worker = TrialWorker(URI, DATABASE, 0, None)
        worker.max_retry = 0
        worker.timeout = 1
        worker.run()

    run_hpos(sum(namespaces.values(), []))

    data = defaultdict(dict)
    hpos_ready, remainings = fetch_hpos_valid_curves(client, namespaces, ['e'], data)
    assert len(remainings) == 0
    assert len(hpos_ready) == 2

    data = consolidate_results(data)

    save_results('test', data, '.')

    loaded_data = load_results('test', '.')

    assert len(loaded_data) == 2
    assert loaded_data['hpo-1'] == data['hpo-1']
    assert loaded_data['hpo-2'] == data['hpo-2']
