import pytest
import logging
import os
from olympus.utils.log import set_log_level

set_log_level(logging.DEBUG)

from olympus.hpo.worker import HPOWorkGroup, make_remote_call
from olympus.utils.testing import my_trial
from olympus.hpo.fidelity import Fidelity
from olympus.hpo import HPOptimizer


FIDELITY = Fidelity(1, 30, 10).to_dict()


def run_nomad_hpo(hpo_name, uri, launch_server=True, fidelity=FIDELITY, group='nomad', workers=10):
    """Worker are converted to HPO when new trials are needed then killed"""
    with HPOWorkGroup(uri, 'olympus', f'classification-{group}-1', clean=True, launch_server=launch_server) as group:
        group.launch_workers(workers)

        params = {
            'a': 'uniform(0, 1)',
            'b': 'uniform(0, 1)',
            'c': 'uniform(0, 1)',
            'lr': 'uniform(0, 1)'
        }

        group.queue_hpo(
            make_remote_call(HPOptimizer, hpo_name, count=30, fidelity=fidelity, space=params),
            make_remote_call(my_trial)
        )

        # wait for workers to do their work
        group.wait()

        if launch_server:
            group.archive('data.zip')

    if launch_server:
        os.remove('data.zip')


def run_master_hpo(hpo_name, uri):
    """HPO is in the main process, works along side the workers"""
    params = {
        'a': 'uniform(0, 1)',
        'b': 'uniform(0, 1)',
        'c': 'uniform(0, 1)',
        'lr': 'uniform(0, 1)'
    }

    hpo = HPOptimizer(hpo_name, count=30, fidelity=FIDELITY, space=params)

    with HPOWorkGroup(uri, 'olympus', 'classification-master-1', clean=True, launch_server=True) as group:
        group.launch_workers(10)

        group.run_hpo(hpo, my_trial)

        group.wait()
        group.archive('data.zip')

    os.remove('data.zip')


def test_nomad():
    from olympus.utils.network import get_free_port

    resources = [
        # 'cockroach://0.0.0.0:{}',
        'mongo://0.0.0.0:{}'
    ]
    optimizers = ['hyperband', 'random_search']

    for hpo in optimizers:
        for uri in resources:
            uri = uri.format(get_free_port())
            run_nomad_hpo(hpo, uri)


def test_master():
    from olympus.utils.network import get_free_port

    resources = [
        # 'cockroach://0.0.0.0:{}',
        'mongo://0.0.0.0:{}'
     ]
    optimizers = ['hyperband', 'random_search']

    for hpo in optimizers:
        for uri in resources:
            uri = uri.format(get_free_port())
            run_master_hpo(hpo, uri)


if __name__ == '__main__':
    # test_master_hpo('cockroach://0.0.0.0:8123', clear=False)
    # test_nomad_rs_hpo('mongo://0.0.0.0:8123', clear=False)
    # test_master()
    import multiprocessing
    from olympus.hpo import Fidelity

    workers = []
    for i in range(10):
        p = multiprocessing.Process(
            target=run_nomad_hpo,
            args=('hyperband', 'mongo://127.0.0.1:27017', False),
            kwargs=dict(fidelity=Fidelity(1, 30, 2).to_dict(), group=f'nomad{i}', workers=10 + i))
        p.start()
        workers.append(p)

    for w in workers:
        w.join()

