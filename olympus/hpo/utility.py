from collections import OrderedDict

from olympus.hpo.fidelity import Fidelity


class FunctionWithSpace:
    def __init__(self, fun, fidelity, **kwargs):
        self.fun = fun
        self.space = OrderedDict(kwargs.items())
        self.fidelity = fidelity

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)


def hyperparameters(**kwargs):
    """Annotates a function with its hyper parameter space

    Examples
    --------

    >>> @hyperparameters(epoch='fidelity(1, 30, 4)', lr='uniform(0, 1)', b='uniform(0, 1)', c='uniform(0, 1)')
    ... def my_trial(epoch, lr, a, b, c, **kwargs):
    ...     return lr * a - b * c

    >>> print(my_trial.space)
    OrderedDict([('lr', 'uniform(0, 1)'), ('b', 'uniform(0, 1)'), ('c', 'uniform(0, 1)')])

    >>> print(my_trial.fidelity)
    fidelity(1, 30, 4)

    """

    def parse_fidelity(fidelity_str):
        idx = fidelity_str.find('(') + 1
        args = fidelity_str[idx:-1]
        return [int(v) for v in args.split(',')]

    def call(f):
        fid = None
        fidelity_key = None

        for k, v in kwargs.items():
            if v.startswith('fidelity'):
                args = parse_fidelity(v)
                fid = Fidelity(*args, name=k)
                fidelity_key = k

        if fidelity_key is not None:
            kwargs.pop(fidelity_key)

        fun_with_space = FunctionWithSpace(f, fid, **kwargs)
        return fun_with_space

    return call
