import copy
from collections import defaultdict

import numpy
import xarray

import pymongo

from sspace.space import compute_identity

from msgqueue.backends import new_client

from olympus.studies.hpo.main import (
    generate_grid_search, generate_noisy_grid_search, generate_random_search,
    generate_bayesopt, generate_hpos, register_hpo, register_hpos, env,
    fetch_hpos_valid_curves, consolidate_results, save_results, load_results)
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
        'd': 'uniform(lower=-1, upper=1)',
        'epoch': 'uniform(lower=1, upper=10)'
    }
}
DEFAULTS = {}


def foo(uid, a, b, c, d, e=1, epoch=0, experiment_name=NAMESPACE,
        client=None):
    result = a + 2 * b - c ** 2 + d + e
    for i in range(epoch + 1):
        data = {'obj': i + result, 'valid': i + result, 'uid': uid, 'epoch': i}
        client.push(METRIC_QUEUE, experiment_name, data, mtype=METRIC_ITEM)
    print(i, data, NAMESPACE)
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


@pytest.mark.skip(reason='Requires integration of RoBO')
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
        assert configs[i]['name'] == 'bayesopt'
        assert configs[i]['space'] == search_space
        assert configs[i]['seed'] == i
        assert configs[i]['namespace'] == f'bayesopt-s-{i}'
        assert configs[i].pop('uid') == compute_identity(configs[i], 16)


def test_generate_hpos():
    hpos = ['grid_search', 'nudged_grid_search', 'noisy_grid_search', 'random_search']
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

    configs = generate_hpos(range(num_experiments), hpos, budget, fidelity, search_space)

    assert len(configs) == len(hpos)
    assert len(configs['grid_search']) == 1
    assert len(configs['nudged_grid_search']) == 1
    assert configs['grid_search'][0]['name'] == 'grid_search'
    assert configs['grid_search'][0].get('nudge') is None
    assert configs['nudged_grid_search'][0]['name'] == 'grid_search'
    assert configs['nudged_grid_search'][0]['nudge'] == 0.5
    for hpo in hpos[2:]:
        for i in range(num_experiments):
            assert configs[hpo][i]['name'] == hpo
            assert configs[hpo][i]['seed'] == i


@pytest.mark.usefixtures('clean_mongodb')
def test_register_hpos(client):
    namespace = 'test-hpo'
    hpos = ['grid_search', 'nudged_grid_search', 'noisy_grid_search', 'random_search'] 
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

    configs = generate_hpos(range(num_experiments), hpos, budget, fidelity, search_space)

    assert client.monitor().read_count(WORK_QUEUE, namespace, mtype=HPO_ITEM) == 0
    new_namespaces = register_hpos(client, namespace, foo, configs, defaults)
    assert len(set(new_namespaces)) == len(configs)
    for hpo, hpo_namespaces in new_namespaces.items():
        for i, hpo_namespace in enumerate(hpo_namespaces):
            messages = client.monitor().messages(WORK_QUEUE, hpo_namespace, mtype=HPO_ITEM)
            assert len(messages) == 1
            assert messages[0].message['hpo']['kwargs'] == configs[hpo][i]


@pytest.mark.usefixtures('clean_mongodb')
def test_register_hpos_resume(client, monkeypatch):
    namespace = 'test-hpo'
    hpos = ['grid_search', 'nudged_grid_search', 'noisy_grid_search', 'random_search'] 
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

    configs = generate_hpos(range(num_experiments), hpos, budget, fidelity, search_space)

    assert client.monitor().read_count(WORK_QUEUE, namespace, mtype=HPO_ITEM) == 0
    new_namespaces = register_hpos(client, namespace, foo, configs, defaults)
    assert len(set(new_namespaces)) == len(configs)

    more_configs = generate_hpos(range(num_experiments + 2), hpos, budget, fidelity, search_space)

    # Save new namespaces for test
    new_namespaces = defaultdict(list)
    def mock_register_hpo(client, namespace, function, config, defaults):
        new_namespaces[config['name']].append(namespace)
        return register_hpo(client, namespace, function, config, defaults)
    monkeypatch.setattr('olympus.studies.hpo.main.register_hpo', mock_register_hpo)
    namespaces = register_hpos(client, namespace, foo, more_configs, defaults)
    assert (len(set(sum(new_namespaces.values(), []))) == 
            len(sum(more_configs.values(), [])) - len(sum(configs.values(), [])))

    # Verify new registered configs
    for hpo, configs in more_configs.items():
        for config in configs:
            messages = client.monitor().messages(WORK_QUEUE, env(namespace, config['namespace']),
                                                 mtype=HPO_ITEM)
            assert len(messages) == 1
            assert messages[0].message['hpo']['kwargs'] == config


@pytest.mark.usefixtures('clean_mongodb')
def test_fetch_hpos_valid_results_first_time(client):
    config = copy.deepcopy(CONFIG)
    num_trials = 5
    config['count'] = num_trials
    config['fidelity'] = Fidelity(0, 0, name='epoch').to_dict()

    register_hpo(client, NAMESPACE + '1', foo, config, {'e': 2})
    register_hpo(client, NAMESPACE + '2', foo, config, {'e': 2})

    worker = TrialWorker(URI, DATABASE, 0, None)
    worker.max_retry = 0
    worker.timeout = 1
    worker.run()

    namespaces = {'hpo' + str(i): [NAMESPACE + str(i)] for i in range(1, 3)}

    data = defaultdict(list)
    remainings = fetch_hpos_valid_curves(client, namespaces, ['e'], data)

    assert len(data) == 2
    assert len(data['hpo1']) == 1
    assert len(data['hpo2']) == 1

    assert data['hpo1'][0].attrs['namespace'] == f'{NAMESPACE}1'
    assert data['hpo1'][0].epoch.values.tolist() == [0]
    assert data['hpo1'][0].order.values.tolist() == list(range(num_trials))
    assert data['hpo1'][0].seed.values.tolist() == [1]
    assert data['hpo1'][0].params.values.tolist() == list('abcd')
    assert data['hpo1'][0].noise.values.tolist() == ['e']
    assert data['hpo1'][0].obj.shape == (1, num_trials, 1)
    assert data['hpo1'][0].valid.shape == (1, num_trials, 1)

    assert data['hpo1'][0] == data['hpo2'][0]


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

    data = defaultdict(list)
    remainings = fetch_hpos_valid_curves(client, namespaces, ['e'], data)
    assert len(remainings) == 2
    assert len(remainings['hpo-1']) == 1
    assert len(remainings['hpo-2']) == 2

    assert len(data) == 1
    assert len(data['hpo-1']) == 1

    assert data['hpo-1'][0].attrs['namespace'] == f'{NAMESPACE}-1-1'

    run_hpos([namespaces['hpo-1'][1], namespaces['hpo-2'][0]])

    remainings = fetch_hpos_valid_curves(client, remainings, ['e'], data)
    assert len(remainings) == 1
    assert len(remainings['hpo-2']) == 1

    assert len(data) == 2
    assert len(data['hpo-1']) == 2
    assert len(data['hpo-2']) == 1

    assert data['hpo-1'][1].attrs['namespace'] == f'{NAMESPACE}-1-2'
    assert data['hpo-2'][0].attrs['namespace'] == f'{NAMESPACE}-2-1'

    run_hpos([namespaces['hpo-2'][1]])

    remainings = fetch_hpos_valid_curves(client, remainings, ['e'], data)
    assert len(remainings) == 0

    assert len(data) == 2
    assert len(data['hpo-1']) == 2
    assert len(data['hpo-2']) == 2

    assert data['hpo-2'][1].attrs['namespace'] == f'{NAMESPACE}-2-2'


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

    data = defaultdict(list)
    remainings = fetch_hpos_valid_curves(client, namespaces, ['e'], data)
    assert len(remainings) == 0

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

    data = defaultdict(list)
    remainings = fetch_hpos_valid_curves(client, namespaces, ['e'], data)
    assert len(remainings) == 0

    data = consolidate_results(data)

    save_results('test', data, '.')

    loaded_data = load_results('test', '.')

    assert len(loaded_data) == 2
    assert loaded_data['hpo-1'] == data['hpo-1']
    assert loaded_data['hpo-2'] == data['hpo-2']
