from msgqueue.backends import new_client

import numpy
import pymongo
import pytest

from olympus.hpo.fidelity import Fidelity
from olympus.hpo import HPOptimizer
from olympus.hpo.parallel import make_remote_call, RESULT_QUEUE, WORK_QUEUE, WORK_ITEM, HPO_ITEM
from olympus.hpo.worker import TrialWorker


URI = 'mongo://127.0.0.1:27017'
DATABASE = 'olympus'


def branin(x, y, epoch=0, uid=None, other=None, experiment_name=None, client=None):
    b = (5.1 / (4.*numpy.pi**2))
    c = (5. / numpy.pi)
    t = (1. / (8.*numpy.pi))
    return 1.*(y-b*x**2+c*x-6.)**2+10.*(1-t)*numpy.cos(x)+10.


def build_robo(model_type, n_init=2, count=5):
    params = {
        'x': 'uniform(-5, 10)',
        'y': 'uniform(0, 15)'
    }

    return HPOptimizer('robo', fidelity=Fidelity(0, 100, 10, 'epoch').to_dict(), space=params,
                       model_type=model_type, count=count, n_init=n_init)


@pytest.fixture
def clean_mongodb():
    client = pymongo.MongoClient(URI.replace('mongo', 'mongodb'))
    client[DATABASE][WORK_QUEUE].drop()
    client[DATABASE][RESULT_QUEUE].drop()


@pytest.mark.parametrize('model_type', ['gp', 'gp_mcmc'])
@pytest.mark.usefixtures('clean_mongodb')
def test_hpo_serializable(model_type):
    namespace = 'test-robo-' + model_type
    n_init = 2
    count = 10

    # First run using a remote worker where serialization is necessary
    # and for which hpo is resumed between each braning call
    hpo = build_robo(model_type, n_init=n_init, count=count)

    namespace = 'test_hpo_serializable'
    hpo = {
        'hpo': make_remote_call(HPOptimizer, **hpo.kwargs),
        'hpo_state': None,
        'work': make_remote_call(branin),
        'experiment': namespace
    }
    client = new_client(URI, DATABASE)
    client.push(WORK_QUEUE, namespace, message=hpo, mtype=HPO_ITEM)
    worker = TrialWorker(URI, DATABASE, 0, None)
    worker.max_retry = 0
    worker.timeout = 1
    worker.run()

    messages = client.monitor().unread_messages(RESULT_QUEUE, namespace)
    for m in messages:
        if m.mtype == HPO_ITEM:
            break

    assert m.mtype == HPO_ITEM, 'HPO not completed'
    worker_hpo = build_robo(model_type)
    worker_hpo.load_state_dict(m.message['hpo_state'])
    assert len(worker_hpo.trials) == count

    # Then run locally where BO is not resumed
    local_hpo = build_robo(model_type, n_init=n_init, count=count)
    i = 0
    best = float('inf')
    while local_hpo.remaining() and i < local_hpo.hpo.count:
        samples = local_hpo.suggest()
        for sample in samples:
            z = branin(**sample)
            local_hpo.observe(sample['uid'], z)
            best = min(z, best)
            i += 1

    assert i == local_hpo.hpo.count

    # Although remote worker was resumed many times, it should give the same
    # results as the local one which was executed in a single run.
    assert worker_hpo.trials == local_hpo.trials
