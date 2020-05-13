from typing import List
from dataclasses import dataclass, field

from sspace import Space
from olympus.hpo.fidelity import Fidelity
from olympus.hpo.optimizer import HyperParameterOptimizer, Trial, LogicError, WaitingForTrials, OptimizationIsDone
from olympus.utils import new_seed, compress_dict, decompress_dict


@dataclass
class _Bracket:
    trials: List[Trial] = field(default_factory=list)
    rung: int = 1

    def append(self, t):
        self.trials.append(t)

    def is_rung_over(self):
        # check that the rung is over
        for trial in self.trials:
            # for rung=n we need n results
            if len(trial.objectives) < self.rung:
                return False

        return True

    def count_remaining(self):
        remaining = 0

        for trial in self.trials:
            # for rung=n we need n results
            if len(trial.objectives) < self.rung:
                remaining += 1

        return remaining

    def promote(self, count):
        assert self.is_rung_over(), 'Rung need to be over to promote'
        self.trials.sort(key=lambda t: t.objective)

        promoted = []
        for i in range(count):
            promoted.append(self.trials[i])

        self.rung += 1
        self.trials = promoted
        return promoted

    def to_dict(self):
        return {
            'trials': [k.uid for k in self.trials],
            'rung': self.rung
        }

    def load_state_dict(self, state, trials):
        self.trials = [trials[k] for k in state['trials']]
        self.rung = state['rung']
        return self


class Hyperband(HyperParameterOptimizer):
    """Hyperband works by removing successively removing half of the worst trials periodically until
    only a few remains, by doing so it does not waste resources training badly performing configurations and
    it favors configurations that train quickly.

    This can cause issue if the best configurations are a slow learners and quick learners start to plateau.

    Parameters
    ----------
    fidelity: Fidelity
        used to generate fidelity budget.
        ``Fidelity.min`` can be used to create a grace period during which no trials are removed from the optimization.
        This will shift all the fidelity by the grace period up to the max fidelity.

    Notes
    -----
    The performance of hyperband is dependent on when the configurations are killed.
    If it happens too soon it might remove good configuration that had a slower start.
    To mitigate this issue you can specify a grace period using ``Fidelity.min``.
    While increasing the grace period will improve performance it will also increase the total number
    of epoch to run.

    The red paths highlight the configurations that have survived up to the last round.
    The gray ones are  the paths that have been killed early.

    .. image:: ../../docs/_static/hpo/hyperband_vanilla.png
        :width: 45 %

    .. image:: ../../docs/_static/hpo/hyperband_grace.png
        :width: 45 %

    Work schedule of Hyperband with 10 workers with ``fidelity=Fidelity(1, 30, base=2)``

    .. image:: ../../docs/_static/hpo/hyperband.png

    Visualization of Hyperband space exploration
    Promotion have been kept to highlight how hyperband picks configuration.

    .. code-block:: python

        space = {
            'a': 'uniform(0, 1)',
            'b': 'uniform(0, 1)',
            'c': 'uniform(0, 1)',
            'lr': 'uniform(0, 1)'
        }

    .. image:: ../../docs/_static/hpo/hyperband_space.png


    References
    ----------
    .. [1] Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, Ameet Talwalkar,
        "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"
    """
    def __init__(self, fidelity: Fidelity, space: Space, seed: int = 0, **kwargs):
        super(Hyperband, self).__init__(fidelity, space, seed, **kwargs)
        self.brackets: List[_Bracket] = []
        self.offset = 0

    @property
    def budget(self):
        # Fidelity(0, 1000, 10, 'epochs')
        # [(300, 10), (30, 100), (3, 1000)]
        # [(30, 100), (3, 1000)]
        # [(3, 1000)]
        # trials: 300 + 30 + 3
        return self.compute_budgets(self.fidelity.max, self.fidelity.base)

    def is_done(self):
        if len(self.brackets) != len(self.budget):
            return False

        for bracket, budget in zip(self.brackets, self.budget):
            if bracket.rung < len(budget):
                return False

        return True

    def max_trials(self):
        return sum([b[0][0] for b in self.budget])

    def suggest(self, **variables):
        # Need to sample the configuration
        if len(self.trials) == 0:
            trials = self.sample(self.max_trials(), **variables)
            return trials

        if self.is_done():
            raise OptimizationIsDone()

        promotions = self.promote()
        if len(promotions) == 0:
            raise WaitingForTrials()

        return promotions

    def new_trials(self, trials):
        start = 0
        for budget in self.budget:
            trial_count, epoch = budget[0]

            self.offset = self.fidelity.min
            epoch = max(epoch, self.fidelity.min)

            bracket = _Bracket()
            self.brackets.append(bracket)

            if start + trial_count > len(trials):
                raise LogicError('Internal Error: Should sample enough for hyperband')

            # fill this bracket with trials
            for trial in trials[start:start + trial_count]:
                trial.params[self.fidelity.name] = epoch
                bracket.append(trial)

            start += trial_count

    def promote(self):
        promotions = []

        for bracket, budget in zip(self.brackets, self.budget):
            if bracket.rung >= len(budget):
                continue

            # is the rung over
            if not bracket.is_rung_over():
                continue

            # we can promote
            trial_count, epoch = budget[bracket.rung]
            promoted = bracket.promote(trial_count)

            for trial in promoted:
                trial.params[self.fidelity.name] = min(epoch + self.offset, self.fidelity.max)
                promotions.append(trial.params)

        return promotions

    @staticmethod
    def compute_budgets(max_resources, reduction_factor):
        """Compute the budgets used for each execution of hyperband"""
        import numpy

        num_brackets = int(numpy.log(max_resources) / numpy.log(reduction_factor))
        B = (num_brackets + 1) * max_resources
        budgets = []

        for bracket_id in range(0, num_brackets + 1):
            bracket_budgets = []
            num_trials = B / max_resources * reduction_factor ** (num_brackets - bracket_id)
            min_resources = max_resources / reduction_factor ** (num_brackets - bracket_id)

            for i in range(0, num_brackets - bracket_id + 1):
                n_i = int(num_trials / reduction_factor ** i)
                min_i = int(numpy.ceil(min_resources * reduction_factor ** i))
                bracket_budgets.append((n_i, min_i))

            budgets.append(bracket_budgets)

        return budgets

    def state_dict(self, compressed=True):
        state = super(Hyperband, self).state_dict(compressed=False)
        state['brackets'] = [b.to_dict() for b in self.brackets]
        state['offset'] = self.offset

        if compressed:
            state = compress_dict(state)

        return state

    @staticmethod
    def from_dict(state):
        state = decompress_dict(state)

        hpo = Hyperband(state['fidelity'], state['space'], state['seed'])
        hpo.load_state_dict(state)
        return hpo

    def load_state_dict(self, state):
        state = decompress_dict(state)

        super(Hyperband, self).load_state_dict(state)
        self.offset = state['offset']
        self.brackets = [
            _Bracket().load_state_dict(b, self.trials) for b in state['brackets']
        ]
        return self

    def info(self):
        return {
            'unique_samples': self.max_trials(),
            'total_epochs': self._total_epochs(),
            'parallelism': self._parallelism()
        }

    def _total_epochs(self):
        epochs = 0

        for bracket in self.budget:
            prev = 0

            for trial, epoch in bracket:
                epochs += trial * (epoch - prev)
                prev = epoch

        return epochs

    def _parallelism(self):
        avg = 0
        count = 0

        for bracket in self.budget:
            bracket_avg = 0
            bracket_count = 0

            for trial, epoch in bracket:
                bracket_avg = trial * epoch
                bracket_count += epoch

            avg += bracket_avg / bracket_count
            count += 1

        return avg / count

    def remaining(self):
        # this is not accurate but the worker requirement lowers through time so this should give us an upper bound

        # compute the number of trials required per rung
        # this takes into account future rungs
        remaining = 0
        for bracket, budget in zip(self.brackets, self.budget):
            if bracket.rung < len(budget):
                trial_count, _ = budget[bracket.rung]
                remaining += trial_count

        # this compute the remaining trials for the current rungs
        # this does not take into account future rungs
        # this takes into account missing results
        remaining2 = 0
        for b in self.brackets:
            remaining2 += b.count_remaining()

        return max(remaining2, remaining)


builders = {
    'hyperband': Hyperband,
}
