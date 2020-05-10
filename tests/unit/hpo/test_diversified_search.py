from olympus.hpo import Fidelity
from olympus.hpo.diversified_search import DiversifiedSearch


def test_check_diversified_search():
    space = {
        'a': 'uniform(0, 1)',
        'b': 'uniform(0, 1)',
    }

    def add(uid, epoch, a, b):
        return a + b

    hpo = DiversifiedSearch(Fidelity(0, 30, name='epoch'), space)

    print(hpo.budget)

    while not hpo.is_done():
        for args in hpo:
            epoch = args['epoch']

            for e in range(epoch):
                r = add(**args)
                args['epoch'] = e + 1
                hpo.observe(args, r)

    for p in hpo.result():
        print(p)

