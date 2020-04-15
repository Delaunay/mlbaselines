import copy
from collections import OrderedDict
from typing import Tuple, List

from olympus.utils import debug
from olympus.utils.functional import select
from olympus.hpo.fidelity import Fidelity

from orion.algo.base import OptimizationAlgorithm
from orion.algo.space import Fidelity as OrionFidelity
from orion.core.io.space_builder import SpaceBuilder


Argument = Tuple[str, 'value']


class HPOptimizer:
    def __init__(self, space, fidelity, hpo='hyperband', **hpo_config):
        hpo = hpo_config.pop('hpo_name', hpo)

        if isinstance(fidelity, dict):
            fidelity = Fidelity.from_dict(fidelity)

        self.optimizer = HPOAdapter(
            hpo,
            space,
            fidelity,
            **hpo_config)

    @staticmethod
    def from_function(function, hpo='hyperband', **hpo_config):
        assert hasattr(function, 'space'), 'Function should be annotated with hyperparameters'
        assert hasattr(function, 'fidelity'), 'Function should have a fidelity parameters'

        self = HPOptimizer(
            function.space, function.fidelity, hpo, **hpo_config)

        return self

    def load_state_dict(self, data):
        self.optimizer = HPOAdapter(**data)
        return self

    def state_dict(self):
        return self.optimizer.state_dict()

    def suggest(self):
        return self.optimizer.suggest()

    def observe(self, results):
        return self.optimizer.observe(results)

    def is_done(self):
        return self.optimizer.is_done

    @property
    def budget(self):
        return self.optimizer.budget

    @property
    def results(self):
        return self.optimizer.results


class HPOAdapter:
    """Make Orion HPO resume-able"""

    def __init__(self, hpo_name, space, fidelity, *,
                 uids=None, results=None, running=None, instance=0, allow_dup=False, **hpo_config):
        # Force consistency of ordering
        if isinstance(space, (dict, OrderedDict)):
            space = list(space.items())

        space.sort(key=lambda i: i[0])

        self.optimizer_name = hpo_name
        self.optimizer_config = hpo_config

        self.fidelity = fidelity
        self.space = OrderedDict(space)

        # remove fidelity from space
        for k, v in self.space.items():
            if v.startswith('fidelity'):
                self.space.pop(k, None)
                break

        # build space with fidelity for Orion
        self.hpo_space = self._build_hpo_space(self.space, fidelity)

        self.uid_set = set(select(uids, set()))
        self.results = select(results, [])
        self.running = select(running, dict())
        self.allow_duplicate = allow_dup
        self.instance = instance + 1

        self.optimizer = OptimizationAlgorithm(
            hpo_name,
            space=self.hpo_space,
            **hpo_config)

        self._resume()
        self.budget = [
            dict(epoch=epoch, trial_count=trial_count) for trial_count, epoch in self._budget[0]
        ]

    def _build_hpo_space(self, space, fidelity):
        temp_space = copy.deepcopy(space)
        temp_space[fidelity.name] = str(fidelity)
        builder = SpaceBuilder()
        return builder.build(temp_space)

    def _resume(self):
        # Observe old trials
        for uid, params, objective in self.results:
            self._observe(params, objective)

        # re accept currently running trials
        for uid, params in self.running.items():
            self._accept(params)

    @property
    def trial_count(self):
        return sum([data['trial_count'] for data in self.budget])

    @property
    def trial_count_remaining(self):
        return self.trial_count - (len(self.running) + len(self.results))

    def state_dict(self):
        return {
            'space': list(self.space.items()),
            'uids': list(self.uid_set),
            'results': self.results,
            'running': self.running,
            'instance': self.instance,
            'allow_dup': self.allow_duplicate,
            'hpo_name': self.optimizer_name,
            'hpo_config': self.optimizer_config,
            'fidelity': self.fidelity.to_dict()
        }

    def suggest(self) -> List[Tuple['uid', List[Argument]]]:
        """Sample new trials or promote old trials"""
        suggestions = []
        new_suggestions = self._suggest()

        while len(new_suggestions) > 0:
            for uid, params in new_suggestions:
                if self.allow_duplicate or uid not in self.uid_set:
                    self._accept(params)
                    self.uid_set.add(uid)

                    # The trial is already running
                    if uid not in self.running:
                        suggestions.append((uid, params))
                        self.running[uid] = params
                    else:
                        debug(f'refusing {uid} because it is already running')
                else:
                    debug(f'refusing because (allow_dup: {self.allow_duplicate})')
            new_suggestions = self._suggest()

        # first call to suggest should have allow_duplicate == False
        # Subsequent call should have it == True
        if not self.allow_duplicate:
            self.allow_duplicate = True

        return suggestions

    def observe(self, results: List[Tuple['uid', List[Argument], float]]):
        """Observe new trials"""
        for uid, params, objective in results:
            assert self.unique_id(self.params_to_points(params)) == uid, 'uid should be deterministic'

            self._observe(params, objective)
            # cant be pop because in case of twin killer we receive more that one result
            self.running.pop(uid, None)
            self.results.append((uid, params, objective))

    def unique_id(self, points):
        """Compute a unique id from the hyper parameters without fidelity"""
        import hashlib
        sh256 = hashlib.sha256()
        try:
            fidelity_dim = self.optimizer.fidelity_index
        # TwinKiller does not have fidelity_index but we know it is the last
        except AttributeError:
            fidelity_dim = len(points) - 1

        for i, p in enumerate(points):

            if i != fidelity_dim:
                sh256.update(str(p).encode('utf8'))

        return sh256.hexdigest()[0:16]

    # @property
    # def _rung_sizes(self):
    #     reduction_factor = self.fidelity.base
    #     return compute_rung_sizes(reduction_factor, len(self._budget))

    @property
    def _budget(self):
        return [[]]

    #     from orion.algo.hyperband.hyperband import compute_budgets
    #
    #     # min_resources = self.fidelity.low
    #     max_resources = self.fidelity.max
    #     reduction_factor = self.fidelity.base
    #
    #     return compute_budgets(max_resources, reduction_factor)

    def convert(self, val):
        import numpy as np
        if isinstance(val, np.int64):
            return int(val)

        if isinstance(val, np.float):
            return float(val)

        return val

    def points_to_params(self, points):
        params = []

        for name, value in zip(self.space.keys(), points):
            params.append((name, self.convert(value)))

        params.append((self.fidelity.name, points[-1]))
        return params

    def params_to_points(self, params):
        points = []
        params_dict = dict(params)

        for k in self.space.keys():
            points.append(params_dict[k])

        return tuple(points) + (params_dict[self.fidelity.name],)

    def _suggest(self):
        points = self.optimizer.suggest()

        if points is None:
            return []

        return [(self.unique_id(p), self.points_to_params(p)) for p in points]

    def _accept(self, params):
        self.optimizer.observe([self.params_to_points(params)], [{'objective': None}])

    def _observe(self, params, value):
        self.optimizer.observe([self.params_to_points(params)], [{'objective': value}])

    @property
    def is_done(self):
        return self.optimizer.is_done


def test_optimizer_wrapper():
    from olympus.hpo.utility import hyperparameters

    @hyperparameters(epoch='fidelity(1, 30, 4)', lr='uniform(0, 1)', b='uniform(0, 1)', c='uniform(0, 1)')
    def my_trial(epoch, lr, **kwargs):
        return lr

    fid = my_trial.fidelity
    space = my_trial.space

    hpo = HPOptimizer(space, fid,
                      hpo='hyperband',
                      seed=1)
    min_val = 100000

    while not hpo.is_done():
        trials = hpo.suggest()

        results = []
        for uid, args in trials:
            result = my_trial(**OrderedDict(args))
            results.append((uid, args, result))
            min_val = min(result, min_val)

        hpo.observe(results)


def test_optimizer_twinkiller():
    import copy
    from olympus.hpo.utility import hyperparameters

    @hyperparameters(epoch='fidelity(1, 30, 4)', lr='uniform(0, 1)', b='uniform(0, 1)', c='uniform(0, 1)')
    def my_trial(epoch, lr, b, **kwargs):
        return [(lr + b) / (e + 1) for e in range(epoch)]

    fid = my_trial.fidelity
    space = my_trial.space

    hpo = HPOptimizer(space, fid,
                      hpo='twinkiller',
                      seed=1)
    min_val = 100000
    last_results = None
    while not hpo.is_done():
        print('=' * 80)
        print('New Round')
        print('Missing: ', hpo.optimizer.optimizer.impl.missing)
        trials = hpo.suggest()

        print(f'Sampled Trial: {trials[0]}')
        print(f'Trials: {len(trials)}')

        results = []
        for uid, args in trials:
            result = my_trial(**OrderedDict(args))
            assert len(result) > 1

            for i, r in enumerate(result):
                new_args = copy.deepcopy(args)
                epoch_arg = list(new_args[-1])
                epoch_arg[1] = i + 1
                new_args[-1] = tuple(epoch_arg)

                results.append((uid, new_args, r))
                min_val = min(min_val, r)

        print('New Results: ', len(results))
        hpo.observe(results)
        last_results = results

    print(hpo.budget)
    print('min is:', min_val)
    for uid, args, objective in last_results:
        print(uid, objective)


if __name__ == '__main__':
    test_optimizer_twinkiller()
    # test_optimizer_wrapper()

