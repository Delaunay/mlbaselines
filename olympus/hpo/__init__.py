from olympus.utils.factory import fetch_factories
from olympus.utils import MissingArgument, warning
from olympus.hpo.fidelity import Fidelity
from olympus.hpo.parallel import ParallelHPO

registered_optimizer = fetch_factories('olympus.hpo', __file__)


def known_hpo():
    return registered_optimizer.keys()


def register_hpo(name, factory, override=False):
    global registered_optimizer

    if name in registered_optimizer:
        warning(f'{name} was already registered, use override=True to ignore')

        if not override:
            return

    registered_optimizer[name] = factory


class RegisteredHPONotFound(Exception):
    pass


class HPOptimizer:
    """Olympus standardized HPO interface

    Examples
    --------

    .. code-block:: python

        def add(a, b):
             return a + b

        hpo = HPOptimizer('hyperband', space={
            'a': 'uniform(0, 1)',
            'b': 'uniform(0, 1)'
        })

        while not hpo.is_done():
            for args in hpo:
                # try a new configuration
                result = add(**args)

                # forward the result to the optimizer
                hpo.observe(args, result)

        hpo.result()
        (OrderedDict([('a', 0.02021839744032572), ('b', 0.05433798833925363), ('uid', 'b6b3c96296beaad9'), ('epoch', 8)]), 0.07455638577957935

    """

    def __init__(self, name=None, *, hpo=None, **kwargs):
        self.hpo = None

        if name:
            hpo_fun = registered_optimizer.get(name)

            if not hpo_fun:
                raise RegisteredHPONotFound(name)

            self.hpo = hpo_fun(**kwargs)

        elif hpo and isinstance(hpo, type):
            self.hpo = hpo(**kwargs)

        else:
            raise MissingArgument('hpo or name need to be set')

        kwargs['name'] = name
        self.kwargs = kwargs

    def insert_manual_sample(self, sample=None, fidelity_override=None, **kwargs):
        return self.hpo.insert_manual_sample(sample, fidelity_override, **kwargs)

    def suggest(self, **kwargs):
        return self.hpo.suggest(**kwargs)

    def observe(self, args, result):
        return self.hpo.observe(args, result)

    def is_done(self):
        return self.hpo.is_done()

    def remaining(self):
        return self.hpo.remaining()

    def load_state_dict(self, state):
        return self.hpo.load_state_dict(state)

    def state_dict(self, compressed=True):
        return self.hpo.state_dict(compressed=compressed)

    def info(self):
        return self.hpo.info()

    def result(self):
        result = sorted(self.hpo.trials.values(), key=lambda x: x.objective)
        return result[0]

    def __iter__(self):
        return iter(self.hpo)

    @property
    def trials(self):
        return self.hpo.trials

    def ctor_call(self):
        return self.kwargs


def check():
    def add(a, b, **kwargs):
        return a + b

    hpo = HPOptimizer('hyperband', fidelity=Fidelity(1, 30, 2), space={
        'a': 'uniform(0, 1)',
        'b': 'uniform(0, 1)'
    })

    while not hpo.is_done():
        for args in hpo:
            # try a new configuration
            result = add(**args)

            # forward the result to the optimizer
            hpo.observe(args, result)

    print(hpo.result())


if __name__ == '__main__':
    check()
