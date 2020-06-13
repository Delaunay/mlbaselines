import copy

import numpy

import pymongo

import pytest

from msgqueue.backends import new_client
from studies import (
    register_hpo,
    get_hpo, fetch_metrics, get_array_names, fetch_hpo_valid_curves,
    save_results, load_results, is_registered, is_hpo_completed)


from olympus.observers.msgtracker import METRIC_QUEUE, METRIC_ITEM
from olympus.hpo.parallel import (
    exec_remote_call, RESULT_QUEUE, WORK_QUEUE, WORK_ITEM, HPO_ITEM)
from olympus.hpo.worker import TrialWorker
from olympus.hpo import Fidelity
from olympus.utils import decompress_dict


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


@pytest.mark.usefixtures('clean_mongodb')
def test_register_hpo_is_actionable(client):
    """Test that the registered HPO have valid workitems and can be executed."""
    namespace = 'test-hpo'
    config = {
        'name': 'random_search',
        'seed': 1, 'count': 1,
        'fidelity': Fidelity(1, 10, name='d').to_dict(),
        'space': {
            'a': 'uniform(-1, 1)',
            'b': 'uniform(-1, 1)',
            'c': 'uniform(-1, 1)',
            'd': 'uniform(-1, 1)'
        }
    }

    defaults = {}
    register_hpo(client, namespace, foo, config, defaults)
    worker = TrialWorker(URI, DATABASE, 0, namespace)
    worker.max_retry = 0
    worker.run()

    assert client.monitor().read_count(WORK_QUEUE, namespace, mtype=WORK_ITEM) == 1
    assert client.monitor().read_count(WORK_QUEUE, namespace, mtype=HPO_ITEM) == 2

    messages = client.monitor().unread_messages(RESULT_QUEUE, namespace, mtype=HPO_ITEM)

    compressed_state = messages[0].message.get('hpo_state')
    assert compressed_state is not None
    state = decompress_dict(compressed_state)

    assert len(state['trials']) == 1
    assert state['trials'][0][1]['objectives'] == [10.715799430116764]


@pytest.mark.usefixtures('clean_mongodb')
def test_is_registered(client):
    assert not is_registered(client, NAMESPACE)

    register_hpo(client, NAMESPACE, foo, CONFIG, DEFAULTS)

    assert is_registered(client, NAMESPACE)


@pytest.mark.usefixtures('clean_mongodb')
def test_is_hpo_completed(client):

    assert not is_hpo_completed(client, NAMESPACE)

    register_hpo(client, NAMESPACE, foo, CONFIG, DEFAULTS)

    assert not is_hpo_completed(client, NAMESPACE)

    worker = TrialWorker(URI, DATABASE, 0, NAMESPACE)
    worker.max_retry = 0
    worker.run()

    assert is_hpo_completed(client, NAMESPACE)


@pytest.mark.usefixtures('clean_mongodb')
def test_get_hpo_non_existant(client):
    with pytest.raises(RuntimeError) as exc:
        get_hpo(client, 'i-dont-exist')

    exc.match(f'No HPO for namespace i-dont-exist or HPO is not completed')


@pytest.mark.usefixtures('clean_mongodb')
def test_get_hpo_non_completed(client):

    register_hpo(client, NAMESPACE, foo, CONFIG, DEFAULTS)
    with pytest.raises(RuntimeError) as exc:
        get_hpo(client, NAMESPACE)

    exc.match(f'No HPO for namespace {NAMESPACE} or HPO is not completed')


@pytest.mark.usefixtures('clean_mongodb')
def test_get_hpo_completed(client):

    register_hpo(client, NAMESPACE, foo, CONFIG, {'e': 2})

    worker = TrialWorker(URI, DATABASE, 0, NAMESPACE)
    worker.max_retry = 0
    worker.run()

    hpo, remote_call = get_hpo(client, NAMESPACE)

    assert len(hpo.trials) == 1
    state_dict = hpo.state_dict(compressed=False)
    assert state_dict['seed'] == CONFIG['seed']
    assert state_dict['fidelity'] == CONFIG['fidelity']
    state_dict['space'].pop('uid')
    assert state_dict['space'] == CONFIG['space']

    # Verify default was passed properly
    assert remote_call['kwargs']['e'] == 2
    remote_call['kwargs'].update(dict(a=1, b=1, c=1, d=1, uid=0, client=client))

    # Verify that the remote_call is indeed callable.
    a = 1
    b = 1
    c = 1
    d = 1
    e = 2
    assert exec_remote_call(remote_call) == a + 2 * b - c ** 2 + d + e


DUPLICATE = 1
SHUFFLED = 2
MISSING_EPOCH = 3
MISSING_FIELD = 4
EMPTY = 5


def populate_metrics(client):
    def log_metric(uid, data):
        data['uid'] = uid
        client.push(METRIC_QUEUE, NAMESPACE, data, mtype=METRIC_ITEM)

    log_metric(DUPLICATE, {'epoch': 0, 'obj': 1})
    log_metric(DUPLICATE, {'epoch': 0, 'obj': 2})
    log_metric(DUPLICATE, {'epoch': 1, 'obj': 3})

    log_metric(SHUFFLED, {'epoch': 1, 'obj': 2})
    log_metric(SHUFFLED, {'epoch': 0, 'obj': 1})
    log_metric(SHUFFLED, {'epoch': 2, 'obj': 3})

    log_metric(MISSING_EPOCH, {'epoch': 0, 'obj': 1})
    log_metric(MISSING_EPOCH, {'epoch': 2, 'obj': 3})

    log_metric(MISSING_FIELD, {'epoch': 0, 'obj': 1})
    log_metric(MISSING_FIELD, {'epoch': 1})
    log_metric(MISSING_FIELD, {'epoch': 2, 'obj': 3})

    log_metric(EMPTY, {})


@pytest.mark.usefixtures('clean_mongodb')
def test_fetch_metrics(client):

    populate_metrics(client)

    metrics = fetch_metrics(client, NAMESPACE)

    assert metrics[DUPLICATE] == [
        {'epoch': 0, 'obj': 1},
        {'epoch': 0, 'obj': 2},
        {'epoch': 1, 'obj': 3}]

    assert metrics[SHUFFLED] == [
        {'epoch': 0, 'obj': 1},
        {'epoch': 1, 'obj': 2},
        {'epoch': 2, 'obj': 3}]

    assert metrics[MISSING_EPOCH] == [
        {'epoch': 0, 'obj': 1},
        {'epoch': 2, 'obj': 3}]

    assert metrics[MISSING_FIELD] == [
        {'epoch': 0, 'obj': 1},
        {'epoch': 1},
        {'epoch': 2, 'obj': 3}]

    assert metrics[EMPTY] == [{}]


def test_get_array_names():
    metrics = dict(
        a=[{'a': 1}, {'e':1}],
        b=[{'b': 1}],
        c=[{'a': 1, 'b': 1}],
        d=[{'a': 1, 'c': 1}],
    )
    names = get_array_names(metrics)
    assert names == set('abce')


@pytest.mark.usefixtures('clean_mongodb')
def test_fetch_hpo_valid_results(client):
    config = copy.deepcopy(CONFIG)
    num_trials = 5
    config['count'] = num_trials

    register_hpo(client, NAMESPACE, foo, config, {'e': 2})
    worker = TrialWorker(URI, DATABASE, 0, NAMESPACE)
    worker.max_retry = 0
    worker.run()

    data = fetch_hpo_valid_curves(client, NAMESPACE, ['e'])

    assert data.attrs['namespace'] == NAMESPACE
    assert data.epoch.values.tolist() == list(range(config['fidelity']['max'] + 1))
    assert data.order.values.tolist() == list(range(num_trials))
    assert data.seed.values.tolist() == [1]
    assert data.params.values.tolist() == list('abcd')
    assert data.noise.values.tolist() == ['e']
    assert data.obj.shape == (config['fidelity']['max'] + 1, num_trials, 1)
    assert numpy.all((data.obj.loc[dict(epoch=10)] - data.obj.loc[dict(epoch=0)]) == 
                     (numpy.ones((num_trials, 1)) * 10))


@pytest.mark.usefixtures('clean_mongodb')
def test_fetch_hpo_valid_results_no_epochs(client):
    config = copy.deepcopy(CONFIG)
    num_trials = 5
    config['count'] = num_trials
    config['fidelity'] = Fidelity(1, 1, name='epoch').to_dict()

    register_hpo(client, NAMESPACE, foo, config, {'e': 2})
    worker = TrialWorker(URI, DATABASE, 0, NAMESPACE)
    worker.max_retry = 0
    worker.run()

    data = fetch_hpo_valid_curves(client, NAMESPACE, ['e'])

    assert data.attrs['namespace'] == NAMESPACE
    assert data.epoch.values.tolist() == [0, 1]
    assert data.order.values.tolist() == list(range(num_trials))
    assert data.seed.values.tolist() == [1]
    assert data.params.values.tolist() == list('abcd')
    assert data.noise.values.tolist() == ['e']
    assert data.obj.shape == (2, num_trials, 1)
    assert data.valid.shape == (2, num_trials, 1)


@pytest.mark.usefixtures('clean_mongodb')
def test_save_load_results(client):
    config = copy.deepcopy(CONFIG)
    num_trials = 2
    config['count'] = num_trials
    config['fidelity'] = Fidelity(1, 1, name='epoch').to_dict()

    register_hpo(client, NAMESPACE, foo, config, {'e': 2})
    worker = TrialWorker(URI, DATABASE, 0, NAMESPACE)
    worker.max_retry = 0
    worker.run()

    data = fetch_hpo_valid_curves(client, NAMESPACE, ['e'])

    save_results(NAMESPACE, data, '.')

    assert load_results(NAMESPACE, '.')
