from sspace import Space
from sspace.space import compute_identity

import orion.algo.base
from orion.algo.grid_search.gridsearch import GridSearch as OrionGridSearch, NoisyGridSearch as OrionNoisyGridSearch

from olympus.hpo.optimizer import Trial, HyperParameterOptimizer, WaitingForTrials, OptimizationIsDone
from olympus.hpo.fidelity import Fidelity
from olympus.utils import new_seed
from olympus.utils.functional import unflatten


class GridSearch(HyperParameterOptimizer):
    """
    """

    def __init__(self, fidelity: Fidelity, space: Space, n_points=5, nudge=None, seed=new_seed(hpo_sampler=0), **kwargs):
        super(GridSearch, self).__init__(fidelity, space, seed, **kwargs)
        self.n_points = n_points
        self.count = n_points ** len(space)
        self.orion_space = self.space.instantiate('Orion')
        self.grid = OrionGridSearch(self.orion_space, n_points=n_points, nudge=nudge).grid

    def sample(self, count=1, **variables):
        samples = []

        submitted_count = len(self.trials)

        for point in self.grid[submitted_count:submitted_count + count]:
            sample = dict(zip(self.orion_space.keys(), point))
            sample.update(variables)
            sample = unflatten(sample)
            sample[self.identity] = compute_identity(sample, self.space._identity_size)
            samples.append(sample)

        self.seed_time += 1
        trials = []

        # Register all the samples
        for s in samples:
            t = Trial(s)
            trials.append(t)
            self.trials[s[self.identity]] = t

        self.new_trials(trials)

        return samples

    def suggest(self, **variables):
        if len(self.trials) == 0:
            return self.sample(self.count, **variables)

        if self.is_done():
            raise OptimizationIsDone()

        raise WaitingForTrials()

    def new_trials(self, trials):
        for trial in trials:
            trial.params[self.fidelity.name] = self.fidelity.max

    def count_done(self):
        count = 0
        for _, trial in self.trials.items():
            if len(trial.objectives) > 0:
                count += 1

        return count

    def is_done(self):
        return self.count_done() == self.count

    def info(self):
        return {
            'unique_samples': self.count,
            'total_epochs': self.fidelity.max * self.count,
            'parallelism': self.count
        }

    def load_state_dict(self, state):
        super(GridSearch, self).load_state_dict(state)
        self.count = state['count']

    def state_dict(self):
        state = super(GridSearch, self).state_dict()
        state['count'] = self.count
        return state

    def remaining(self):
        return self.count - self.count_done()


class NoisyGridSearch(GridSearch):
    """
    """

    def __init__(self, fidelity: Fidelity, space: Space, seed=new_seed(hpo_sampler=0), n_points=5, deltas=None, **kwargs):
        super(NoisyGridSearch, self).__init__(fidelity, space, seed, **kwargs)
        self.n_points = n_points
        self.count = n_points ** len(space)
        self.orion_space = self.space.instantiate('Orion')
        self.grid = OrionNoisyGridSearch(self.orion_space, n_points=n_points, deltas=deltas, seed=seed).grid


builders = {
    'grid_search': GridSearch,
    'noisy_grid_search': NoisyGridSearch
}
