from sspace import Space

from olympus.hpo.optimizer import HyperParameterOptimizer, WaitingForTrials, OptimizationIsDone
from olympus.hpo.fidelity import Fidelity
from olympus.utils import new_seed, compress_dict, decompress_dict


class RandomSearch(HyperParameterOptimizer):
    """Randomly samples sets of hyper parameter and return the best one

    Parameters
    ----------
    count: int
        Number of configuration to sample

    Notes
    -----

    .. image:: ../../docs/_static/hpo/rs_space.png
    """

    def __init__(self, fidelity: Fidelity, count: int, space: Space, seed=new_seed(hpo_sampler=0),
                 pool_size=None, **kwargs):
        super(RandomSearch, self).__init__(fidelity, space, seed, **kwargs)
        self.count = count
        if pool_size is None:
            pool_size = self.count
        self.pool_size = pool_size

    def suggest(self, **variables):
        if len(self.trials) < self.count:
            return self.sample(self.pool_size, **variables)

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
        state = decompress_dict(state)

        super(RandomSearch, self).load_state_dict(state)
        self.count = state['count']

    def state_dict(self, compressed=True):
        state = super(RandomSearch, self).state_dict(compressed=False)
        state['count'] = self.count

        if compressed:
            state = compress_dict(state)

        return state

    def remaining(self):
        return self.count - self.count_done()


builders = {
    'random_search': RandomSearch,
}
