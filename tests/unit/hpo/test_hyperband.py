from sspace import Space

from olympus.hpo.fidelity import Fidelity
from olympus.hpo.optimizer import WaitingForTrials, OptimizationIsDone
from olympus.hpo.hyperband import Hyperband


def test_hyperband_api():
    import random

    params = Space.from_dict({
        'a': 'uniform(0, 1)'
    })

    hpo = Hyperband(Fidelity(0, 1000, 10, 'epochs'), params)
    assert not hpo.is_done()

    for rung in range(3):
        params_set = hpo.suggest()

        for i, params in enumerate(params_set):
            print(i, params)

        try:
            hpo.suggest()
            raise RuntimeError()
        except WaitingForTrials:
            pass
        except OptimizationIsDone:
            pass

        for i, params in enumerate(params_set):
            v = random.uniform(0, 1)
            if i == len(params_set) - 1:
                v = 1e-10

            hpo.observe(params, v)

        print('-------')

    assert hpo.is_done()
    print(hpo.result())
    print(hpo.info())


def test_hyperband_simple_sequential():
    import random

    params = Space.from_dict({
        'a': 'uniform(0, 1)'
    })

    hpo = Hyperband(Fidelity(0, 1000, 10, 'epochs'), params)

    for params in hpo:
        hpo.observe(params, result=random.uniform(0, 1))

    assert hpo.is_done()
    print(hpo.result())
    print(hpo.info())
