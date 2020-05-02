import copy

from msgqueue.backends import new_client
import numpy

import pymongo
import pytest

from sspace import Space

from robo.fmin import bayesian_optimization

from olympus.hpo import HPOptimizer
from olympus.hpo.fidelity import Fidelity
from olympus.hpo.robo import RoBO, build_bounds
from olympus.hpo.random_search import RandomSearch
from olympus.hpo.optimizer import WaitingForTrials
from olympus.hpo.parallel import make_remote_call, RESULT_QUEUE, WORK_QUEUE, WORK_ITEM, HPO_ITEM
from olympus.hpo.worker import TrialWorker


URI = 'mongo://127.0.0.1:27017'
DATABASE = 'olympus'


@pytest.fixture
def clean_mongodb():
    client = pymongo.MongoClient(URI.replace('mongo', 'mongodb'))
    client[DATABASE][WORK_QUEUE].drop()
    client[DATABASE][RESULT_QUEUE].drop()


def branin(x, y, epoch=0, uid=None, other=None, experiment_name=None, client=None):
    b = (5.1 / (4.*numpy.pi**2))
    c = (5. / numpy.pi)
    t = (1. / (8.*numpy.pi))
    return 1.*(y-b*x**2+c*x-6.)**2+10.*(1-t)*numpy.cos(x)+10.


def branin_for_original_robo(x):
    return branin(x[0], x[1])


def get_robo_results(count):
    lower = numpy.array([-5, 0])
    upper = numpy.array([10, 15])
    results = bayesian_optimization(branin_for_original_robo, lower, upper, num_iterations=count)
    return results


def build_robo(model_type, n_init=2, count=5):
    #params = Space.from_dict({
    params = {
        'x': 'uniform(-5, 10)',
        'y': 'uniform(0, 15)'
    }

    return HPOptimizer('robo', fidelity=Fidelity(0, 1000, 10, 'epoch').to_dict(), space=params,
                       model_type=model_type, count=count, n_init=n_init)


@pytest.fixture
def robo():
    return build_robo('gp').hpo


@pytest.fixture
def robo_mcmc():
    return build_robo('gp_mcmc').hpo


def test_optimization(robo_mcmc):
    robo_mcmc.n_init = 3
    robo_mcmc.count = 30
    i = 0
    best = float('inf')
    while robo_mcmc.remaining() and i < robo_mcmc.count:
        samples = robo_mcmc.suggest()
        for sample in samples:
            z = branin(**sample)
            robo_mcmc.observe(sample['uid'], z)
            best = min(z, best)
            i += 1

    assert i == robo_mcmc.count
    assert best < 0.5


def test_resuming_rng_states(robo):
    init_state_dict = robo.state_dict()
    samples = robo.suggest()
    assert len(samples) == robo.n_init
    for i, sample in enumerate(samples):
        robo.observe(sample['uid'], i)
    assert robo.suggest() != samples
    robo.load_state_dict(init_state_dict)
    assert robo.suggest() == samples

    for i, sample in enumerate(samples):
        robo.observe(sample['uid'], i)


@pytest.mark.parametrize('model_type', ['gp', 'gp_mcmc'])
def test_resuming_rng_states_after_init_phase(model_type):
    robo = build_robo(model_type).hpo
    init_samples = robo.suggest()
    assert len(init_samples) == robo.n_init
    for sample in init_samples:
        robo.observe(sample['uid'], branin(**sample))

    # First BO point
    bo_state_dict = robo.state_dict()
    robo_point = robo.robo.choose_next(robo.X, robo.y)
    # RNG changed, not suggesting same point twice.
    assert not numpy.allclose(robo.robo.choose_next(robo.X, robo.y), robo_point)
    robo.load_state_dict(bo_state_dict)
    # RNG restored, suggesting same point again.
    assert numpy.allclose(robo.robo.choose_next(robo.X, robo.y), robo_point)
    
    # Let's try for second point, because then GP_MCMC has p0 and need to keep it properly
    robo.load_state_dict(bo_state_dict)
    samples = robo.suggest()
    assert len(samples) == 1
    robo.observe(samples[0]['uid'], branin(**samples[0]))

    # Second point
    second_state_dict = robo.state_dict()
    robo_point = robo.robo.choose_next(robo.X, robo.y)
    # RNG changed, not suggesting same point twice.
    assert not numpy.allclose(robo.robo.choose_next(robo.X, robo.y), robo_point)
    robo.load_state_dict(second_state_dict)
    # RNG restored, suggesting same point again.
    assert numpy.allclose(robo.robo.choose_next(robo.X, robo.y), robo_point)

    # Let's try for third point as well, because... why not. :)
    robo.load_state_dict(second_state_dict)
    samples = robo.suggest()
    assert len(samples) == 1
    robo.observe(samples[0]['uid'], branin(**samples[0]))

    # Third point
    third_state_dict = robo.state_dict()
    robo_point = robo.robo.choose_next(robo.X, robo.y)
    # RNG changed, not suggesting same point twice.
    assert not numpy.allclose(robo.robo.choose_next(robo.X, robo.y), robo_point)
    robo.load_state_dict(third_state_dict)
    # RNG restored, suggesting same point again.
    assert numpy.allclose(robo.robo.choose_next(robo.X, robo.y), robo_point)


def test_resuming_init_design_not_completed(robo, monkeypatch):
    init_state_dict = robo.state_dict()
    samples = robo.suggest()
    assert len(samples) == robo.n_init
    for sample in samples:
        robo.observe(sample['uid'], branin(**sample))

    # Fake that init did not register all points properly
    robo.n_init = 3

    def should_not_choose_yet(**variables):
        raise RuntimeError('How dare you choose!')

    monkeypatch.setattr(robo, 'choose_next', should_not_choose_yet)

    # Complete init design
    samples = robo.suggest()
    assert len(samples) == 1
    robo.observe(samples[0]['uid'], branin(**samples[0]))

    # Now use BO to choose next point
    with pytest.raises(RuntimeError) as exc:
        robo.suggest()

    assert exc.match('How dare you choose!')


def test_default_variables(robo):
    # In init phase
    samples = robo.suggest(other=6)
    assert len(samples) == robo.n_init
    for sample in samples:
        assert sample['other'] == 6
        robo.observe(sample['uid'], branin(**sample))

    # Now in BO phase
    samples = robo.suggest(other=6)
    assert len(samples) == 1
    assert sample['other'] == 6


def test_waiting_for_trials(robo):
    # Init phase
    samples = robo.suggest()
    assert len(samples) == robo.n_init
    assert robo.is_waiting()
    with pytest.raises(WaitingForTrials):
        robo.suggest()

    for sample in samples:
        robo.observe(sample['uid'], branin(**sample))

    assert not robo.is_waiting()

    # Fake init phase is not completed
    robo.n_init += 1
    assert not robo.is_waiting()

    samples = robo.suggest()
    assert len(samples) == 1
    assert robo.is_waiting()
    with pytest.raises(WaitingForTrials):
        robo.suggest()

    robo.observe(samples[0]['uid'], branin(**samples[0]))

    # Test during BO phase
    for i in range(2):
        assert not robo.is_waiting()
        samples = robo.suggest()
        assert len(samples) == 1
        assert robo.is_waiting()
        with pytest.raises(WaitingForTrials):
            robo.suggest()

        robo.observe(samples[0]['uid'], branin(**samples[0]))

    assert not robo.is_waiting()
    assert robo.is_done()


def test_sample_in_log_space(robo):
    params = Space.from_dict({
        'x': 'uniform(-5, 10)',
        'y': 'loguniform(12, 15)'
    })
    robo = RoBO(Fidelity(0, 1000, 10, 'epoch'), params, count=15, n_init=10)

    while not robo.is_done():
        samples = robo.suggest()
        for sample in samples:
            robo.observe(sample['uid'], branin(**sample))
            assert 12 <= sample['y'] <= 15


def test_X_with_linear_dims(robo):
    params = Space.from_dict({
        'x': 'uniform(-5, 10)',
        'y': 'uniform(0, 15)'
    })
    robo = RoBO(Fidelity(0, 1000, 10, 'epoch'), params, count=5, n_init=2)

    while not robo.is_done():
        samples = robo.suggest()
        for sample in samples:
            robo.observe(sample['uid'], branin(**sample))

    assert robo.X.shape == (5, 2)
    assert robo.y.shape == (5,)
    assert robo.X.min(0)[0] >= -5
    assert robo.X.min(0)[1] >= 0
    assert robo.X.max(0)[0] <= 10
    assert robo.X.max(0)[1] <= 15


def test_X_with_log_dims(robo):
    params = Space.from_dict({
        'x': 'uniform(-5, 10)',
        'y': 'loguniform(1, 15)'
    })
    robo = RoBO(Fidelity(0, 1000, 10, 'epoch'), params, count=5, n_init=2)

    while not robo.is_done():
        samples = robo.suggest()
        for sample in samples:
            robo.observe(sample['uid'], branin(**sample))

    assert robo.X.shape == (5, 2)
    assert robo.y.shape == (5,)
    assert robo.X.min(0)[0] >= -5
    assert robo.X.min(0)[1] >= numpy.log(1)
    assert robo.X.max(0)[0] <= 10
    assert robo.X.max(0)[1] <= numpy.log(15)


def test_build_bounds():
    space = Space.from_dict({
        'x': 'uniform(-5, 10)',
        'y': 'uniform(1, 15)'
    })

    lower, upper = build_bounds(space.instantiate('Orion'))
    assert lower.tolist() == [-5, 1]
    assert upper.tolist() == [10, 15]

    space = Space.from_dict({
        'x': 'uniform(-5, 10)',
        'y': 'loguniform(1, 15)'
    })

    lower, upper = build_bounds(space.instantiate('Orion'))
    assert lower.tolist() == [-5, numpy.log(1)]
    assert upper.tolist() == [10, numpy.log(15)]


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
