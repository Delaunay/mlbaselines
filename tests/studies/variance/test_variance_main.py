import copy

import numpy
import xarray

import pymongo

from sspace.space import compute_identity

from msgqueue.backends import new_client

from olympus.hpo import Fidelity
from olympus.hpo.parallel import WORK_QUEUE, WORK_ITEM
from olympus.hpo.worker import TrialWorker
from olympus.observers.msgtracker import METRIC_QUEUE, METRIC_ITEM
from studies import create_valid_curves_xarray
from studies import (
    generate, fetch_registered, env, register, remaining, fetch_results, get_medians,
    save_results, load_results, create_trials, fetch_vars_stats)

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



def foo(uid, a, b, c, d, e=1, epoch=0,
        experiment_name=NAMESPACE, client=None):
    result = a + 2 * b - c ** 2 + d + e
    client = new_client(URI, DATABASE)
    for i in range(epoch + 1):
        data = {'obj': i + result, 'valid': i + result, 'uid': uid, 'epoch': i}
        client.push(METRIC_QUEUE, experiment_name, data, mtype=METRIC_ITEM)
    return result + i


@pytest.fixture
def clean_mongodb():
    client = pymongo.MongoClient('mongodb://127.0.0.1:27017')
    print('dropping')
    client['olympus'][WORK_QUEUE].drop()


@pytest.fixture
def client():
    return new_client('mongo://127.0.0.1:27017', 'olympus')


def build_data(size):

    # TODO: variables is needed to stack the data for diff variables. For the get_median tests
    #        and others..... 

    epochs = 5
    defaults = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    params = {'c': 2, 'd': 3, 'epoch': epochs}
    variables = 'abc'
    configs = generate(range(size), variables, defaults=defaults, add_reference=False)

    n_vars = len(variables)
    n_seeds = size

    objectives = numpy.arange(n_vars * n_seeds * (epochs + 1))
    numpy.random.RandomState(0).shuffle(objectives)
    objectives = objectives.reshape((epochs + 1, n_seeds, n_vars))

    metrics = dict()
    for var_i, (variable, v_configs) in enumerate(configs.items()):
        for seed_i, config in enumerate(v_configs):
            metrics[config['uid']] = [
                {'epoch': i, 'objective': objectives[i, seed_i, var_i]}
                for i in range(epochs + 1)]

    data = []
    param_names = list(sorted(params.keys()))
    for variable in configs.keys():
        trials = create_trials(configs[variable], params, metrics)
        data.append(
            create_valid_curves_xarray(
                trials, metrics, list(variables), epochs, param_names, seed=variable))

    return xarray.combine_by_coords(data)


def test_generate():
    defaults = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    configs = generate(range(10), 'abc', defaults=defaults, add_reference=False)

    assert list(configs.keys()) == list('abc')

    for name in 'abc':
        assert len(configs[name]) == 10

    def test_doc(name, i):
        a_doc = copy.copy(defaults)
        a_doc[name] = i
        a_doc['_variable'] = name
        a_doc['uid'] = compute_identity(a_doc, 16)
        a_doc.pop('_variable')
        return a_doc

    for i in range(10):
        for name in 'abc':
            assert configs[name][i] == test_doc(name, i)


def test_generate_with_references():
    defaults = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    configs = generate(range(10), 'abc', defaults=defaults, add_reference=True)

    variables = list('abc') + ['reference']

    assert list(configs.keys()) == variables

    for name in variables:
        assert len(configs[name]) == 10

    def test_doc(name, i):
        a_doc = copy.copy(defaults)
        if name == 'reference':
            a_doc['_repetition'] = i
        else:
            a_doc[name] = i
        a_doc['_variable'] = name
        a_doc['uid'] = compute_identity(a_doc, 16)
        if name == 'reference':
            a_doc.pop('_repetition')
        a_doc.pop('_variable')
        return a_doc

    for i in range(10):
        for name in 'abc':
            assert configs[name][i] == test_doc(name, i)



@pytest.mark.usefixtures('clean_mongodb')
def test_fetch_registered(client):
    namespace = 'test'
    variables = list('abc') + ['reference']
    namespaces = [env(namespace, v) for v in variables]

    # test fetch empty
    assert fetch_registered(client, namespaces) == set()

    # insert a couple of calls
    for i, variable in enumerate(variables):
        for j in range(2):
            data = {'name': variable, 'uid': i * 2 + j}
            message = {'kwargs': data}
            client.push(WORK_QUEUE, env(namespace, variable), message, mtype=WORK_ITEM)

    # test fetch
    assert fetch_registered(client, namespaces) == set(range(4 * 2))

    # Simulate one worker reading one message
    workitem = client.dequeue(WORK_QUEUE, env(namespace, variable))
    # And completing it
    client.mark_actioned(WORK_QUEUE, workitem)

    # Should not affect the query
    assert fetch_registered(client, namespaces) == set(range(4 * 2))


@pytest.mark.usefixtures('clean_mongodb')
def test_register_uniques(client):
    defaults = {'a': 1000, 'b': 1001, 'c': 1002, 'd': 3}
    namespace = 'test'
    configs = generate(range(3), 'abc', defaults=defaults)
    variables = list('abc') + ['reference']
    namespaces = [env(namespace, v) for v in variables]

    assert fetch_registered(client, namespaces) == set()
    register(client, foo, namespace, configs)
    assert len(fetch_registered(client, namespaces)) == 4 * 3


@pytest.mark.usefixtures('clean_mongodb')
def test_register_resume(client):
    defaults = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    namespace = 'test'
    configs = generate(range(3), 'abc', defaults=defaults)
    variables = list('abc') + ['reference']
    namespaces = [env(namespace, v) for v in variables]

    assert fetch_registered(client, namespaces) == set()
    new_registered = register(client, foo, namespace, configs)
    assert len(fetch_registered(client, namespaces)) == 4 * 3
    assert fetch_registered(client, namespaces) == new_registered

    # Resume with 10 seeds per configs this time.
    configs = generate(range(10), 'abc', defaults=defaults)
    new_registered = register(client, foo, namespace, configs)
    assert len(fetch_registered(client, namespaces)) == 4 * 3 + 4 * 7
    assert fetch_registered(client, namespaces) != new_registered
    assert len(new_registered) == 4 * 7


@pytest.mark.usefixtures('clean_mongodb')
def test_remaining(client):
    defaults = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    configs = generate(range(1), 'ab', defaults=defaults)
    namespace = 'test'
    register(client, foo, namespace, configs)

    def get_stats(variables):
        trial_stats = fetch_vars_stats(client, namespace)
        print(trial_stats)
        return {v: trial_stats[f'test-{v}'] for v in variables}

    assert remaining(get_stats('ab'))

    assert remaining(get_stats('a'))
    workitem = client.dequeue(WORK_QUEUE, env(namespace, 'a'))
    assert remaining(get_stats('a'))
    client.mark_actioned(WORK_QUEUE, workitem)
    assert not remaining(get_stats('a'))

    assert remaining(get_stats('ab'))
    workitem = client.dequeue(WORK_QUEUE, env(namespace, 'b'))
    assert remaining(get_stats('ab'))
    client.mark_actioned(WORK_QUEUE, workitem)
    assert not remaining(get_stats('ab'))


@pytest.mark.usefixtures('clean_mongodb')
def test_fetch_results_non_completed(client):
    defaults = {'a': 0, 'b': 1}
    params = {'c': 2, 'd': 3}
    defaults = {'epoch': 0}
    medians = ['a']
    configs = generate(range(2), 'ab', params)
    namespace = 'test'
    register(client, foo, namespace, configs)

    with pytest.raises(RuntimeError) as exc:
        fetch_results(client, namespace, configs, medians, params, defaults)

    assert exc.match('Not all trials are completed')


@pytest.mark.usefixtures('clean_mongodb')
def test_fetch_results_corrupt_completed(client):
    defaults = {'a': 0, 'b': 1}
    params = {'c': 2, 'd': 3}
    defaults = {'epoch': 0}
    medians = ['a']
    num_items = 2
    configs = generate(range(num_items), 'ab', defaults=defaults)
    namespace = 'test'
    register(client, foo, namespace, configs)

    for variable in configs.keys():
        for i in range(num_items):
            workitem = client.dequeue(WORK_QUEUE, env(namespace, variable))
            client.mark_actioned(WORK_QUEUE, workitem)

    with pytest.raises(RuntimeError) as exc:
        fetch_results(client, namespace, configs, medians, params, defaults)

    assert exc.match('Nothing found in result queue for trial')


@pytest.mark.usefixtures('clean_mongodb')
def test_fetch_results_all_completed(client):
    defaults = {'a': 1000, 'b': 1001}
    params = {'c': 2, 'd': 3, 'epoch': 5}
    defaults.update(params)
    medians = ['a']
    num_items = 2
    configs = generate(range(num_items), 'ab', defaults=defaults)
    namespace = 'test'
    register(client, foo, namespace, configs)

    print(configs)

    worker = TrialWorker(URI, DATABASE, 0, None)
    worker.max_retry = 0
    worker.timeout = 1
    worker.run()

    print(fetch_vars_stats(client, namespace))

    data = fetch_results(client, namespace, configs, medians, params, defaults)

    assert data.medians == ['a']
    assert data.noise.values.tolist() == ['a', 'b']
    assert data.params.values.tolist() == ['c', 'd']
    assert data.order.values.tolist() == [0, 1]
    assert data.epoch.values.tolist() == list(range(params['epoch'] + 1))
    assert data.uid.shape == (3, 2)
    assert data.seed.values.tolist() == data.noise.values.tolist() + ['reference']
    assert data.a.values.tolist() == [
        [0, 1000, 1000], [1, 1000, 1000]]
    assert data.b.values.tolist() == [
        [1001, 0, 1001], [1001, 1, 1001]]
    assert data.c.values.tolist() == [
        [2, 2, 2], [2, 2, 2]]
    assert data.d.values.tolist() == [
        [3, 3, 3], [3, 3, 3]]

    assert (data.obj.loc[dict(order=0, seed='a')].values.tolist() ==
            list(range(2002, 2002 + params['epoch'] + 1)))


def test_get_medians_odd():
    data = build_data(5)
    median_seeds = get_medians(data, ['a'], 'objective')
    assert median_seeds == {'a': 0}

    median_seeds = get_medians(data, ['a', 'b'], 'objective')
    assert median_seeds == {'a': 0, 'b': 0}

    median_seeds = get_medians(data, ['b', 'c'], 'objective')
    assert median_seeds == {'b': 0, 'c': 3}


def test_get_medians_even():
    data = build_data(4)
    median_seeds = get_medians(data, ['a'], 'objective')
    assert median_seeds == {'a': 1}

    median_seeds = get_medians(data, ['a', 'b'], 'objective')
    assert median_seeds == {'a': 1, 'b': 1}

    median_seeds = get_medians(data, ['b', 'c'], 'objective')
    assert median_seeds == {'b': 1, 'c': 1}


def test_save_load_results():
    data = build_data(4)

    save_results('test', data, '.')

    assert load_results('test', '.') == data
