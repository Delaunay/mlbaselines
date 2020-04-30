import copy

import numpy

import pymongo

import pytest

from msgqueue.backends import new_client

from olympus.observers.msgtracker import MSGQTracker, METRIC_QUEUE, METRIC_ITEM, metric_logger
from olympus.hpo.parallel import (
    exec_remote_call, make_remote_call, RESULT_QUEUE, WORK_QUEUE, WORK_ITEM, HPO_ITEM)
from olympus.hpo.worker import TrialWorker
from olympus.hpo import Fidelity
from olympus.studies.searchspace.main import register_hpo, fetch_hpo_valid_curves
from olympus.studies.searchspace.plot import xarray_to_scipy_results, plot


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
        'd': 'loguniform(lower=1, upper=10)',
    }
}
DEFAULTS = {}


def foo(uid, a, b, c, d, e=1, epoch=0, experiment_name=NAMESPACE, client=None):
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
def test_convert_xarray_to_scipy_results(client):
    config = copy.deepcopy(CONFIG)
    num_trials = 10
    config['count'] = num_trials
    config['fidelity'] = Fidelity(0, 0, name='epoch').to_dict()

    register_hpo(client, NAMESPACE, foo, config, {'e': 2})
    worker = TrialWorker(URI, DATABASE, 0, NAMESPACE)
    worker.max_retry = 0
    worker.run()

    data = fetch_hpo_valid_curves(client, NAMESPACE, ['e'])

    scipy_results = xarray_to_scipy_results(config['space'], 'obj', data)

    assert scipy_results.x[0] == data.a.values[8, 0]
    assert scipy_results.x[1] == data.b.values[8, 0]
    assert scipy_results.x[2] == data.c.values[8, 0]
    assert scipy_results.x[3] == numpy.log(data.d.values[8, 0])
    assert scipy_results.fun == data.obj.values[0, 8, 0]
    assert len(scipy_results.x_iters) == num_trials


@pytest.mark.usefixtures('clean_mongodb')
def test_plot(client):
    config = copy.deepcopy(CONFIG)
    num_trials = 50
    config['count'] = num_trials
    config['fidelity'] = Fidelity(0, 0, name='epoch').to_dict()

    register_hpo(client, NAMESPACE, foo, config, {'e': 2})
    worker = TrialWorker(URI, DATABASE, 0, NAMESPACE)
    worker.max_retry = 0
    worker.run()

    data = fetch_hpo_valid_curves(client, NAMESPACE, ['e'])

    plot(config['space'], 'obj', data, 'test.png')
