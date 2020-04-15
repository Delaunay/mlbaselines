from sspace import Space

from olympus.hpo.optimizer import HyperParameterOptimizer, WaitingForTrials, OptimizationIsDone
from olympus.hpo.fidelity import Fidelity
from olympus.utils import new_seed


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

    def __init__(self, fidelity: Fidelity, count: int, space: Space, seed=new_seed(hpo_sampler=0), **kwargs):
        super(RandomSearch, self).__init__(fidelity, space, seed, **kwargs)
        self.count = count

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
        super(RandomSearch, self).load_state_dict(state)
        self.count = state['count']

    def state_dict(self):
        state = super(RandomSearch, self).state_dict()
        state['count'] = self.count
        return state

    def remaining(self):
        return self.count - self.count_done()


builders = {
    'random_search': RandomSearch,
}
