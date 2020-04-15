from collections import defaultdict

import numpy as np
import base64

from sspace import Space
from olympus.hpo.fidelity import Fidelity
from olympus.hpo.optimizer import HyperParameterOptimizer, WaitingForTrials, OptimizationIsDone
from olympus.hpo.hyperband import Hyperband
from olympus.utils import new_seed


class DiversifiedSearch(HyperParameterOptimizer):
    """Select configuration that are the most different in order to keep our options open

    Parameters
    ----------
    lower_quantile: 0.10
        cut the 10% most similar trials from being selected

    upper_quantile: 0.80
        cut the 20% most different trials from being selected
    """
    def __init__(self, fidelity: Fidelity, space: Space, lower_quantile=0.1, upper_quantile=0.8, seed: int = new_seed(hpo_sampler=0), **kwargs):
        super(DiversifiedSearch, self).__init__(fidelity, space, seed, **kwargs)

        count = self.budget[0]['count']
        epoch = self.budget[0]['epoch']

        self.trajectories = np.zeros((epoch, count))
        self.trial2index = {}
        self.index2trial = {}
        self.round = 0
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def state_dict(self):
        encoded_numpy = base64.b64encode(self.trajectories.tobytes())

        state = super(DiversifiedSearch, self).state_dict()
        state['trajectories'] = encoded_numpy
        state['shape'] = self.trajectories.shape
        state['trial2index'] = self.trial2index
        state['index2trial'] = self.index2trial
        state['round'] = self.round
        state['lower_quantile'] = self.lower_quantile
        state['upper_quantile'] = self.upper_quantile
        return state

    def load_state_dict(self, state):
        super(DiversifiedSearch, self).load_state_dict(state)
        trajectories = np.frombuffer(base64.b64decode(state['trajectories']))
        trajectories.setflags(write=1)

        self.trajectories = trajectories.reshape(state['shape'])
        self.trial2index = self.trial2index
        self.index2trial = self.index2trial
        self.round = self.round
        self.lower_quantile = self.lower_quantile
        self.upper_quantile = self.upper_quantile

    @property
    def budget(self):
        # DS does not have brackets
        budget = Hyperband.compute_budgets(self.fidelity.max, self.fidelity.base)[0]
        return [dict(epoch=b[1], count=b[0]) for b in budget]

    def suggest(self, **variables):
        if len(self.trials) == 0:
            trials = self.sample(self.budget[self.round]['count'], **variables)
            return trials

        if self.is_done():
            raise OptimizationIsDone()

        if self.ready():
            return self.promote()

        raise WaitingForTrials()

    def promote(self):
        selected = self.rank()
        count = len(selected)
        epoch = self.budget[self.round + 1]['epoch']
        assert count == self.budget[self.round + 1]['count']

        old = self.trajectories

        self.trajectories = np.zeros((epoch, count))
        self.trajectories[:, :] = None

        self.trajectories[:old.shape[0], :] = old[:, selected]

        samples = []
        if selected is not None:
            new_idx_trials = {}
            new_trials_idx = {}

            #  fetch the points matching the selected trials
            for new_idx, old_idx in enumerate(selected):
                trial_uid = self.index2trial[old_idx]
                samples.append(self.trials[trial_uid].params)

                new_trials_idx[trial_uid] = new_idx
                new_idx_trials[new_idx] = trial_uid

            # update indexes
            self.index2trial = new_idx_trials
            self.trial2index = new_trials_idx

        self.round += 1
        epoch = self.budget[self.round]['epoch']

        for s in samples:
            s[self.fidelity.name] = epoch

        return samples

    def rank(self):
        end = self.budget[self.round]['epoch'] - 1

        best_trial = np.argmin(self.trajectories[end, :])
        count = self.trajectories.shape[1]

        weights = defaultdict(int)

        for i in range(count):
            if i != best_trial:
                diff = self.trajectories[:, best_trial] - self.trajectories[:, i]
                weights[i] = np.var(diff)

        weights = list(weights.items())
        weights.sort(key=lambda item: item[1])

        selected = self.select_by_quantiles(weights, self.budget[self.round + 1]['count'] - 1)
        selected.append(best_trial)
        return selected

    def select_by_quantiles(self, weights, count):
        # get the number of trials we need for the next round
        percentages = np.linspace(self.lower_quantile, self.upper_quantile, num=count)
        quantiles = np.quantile(
            [w[1] for w in weights],
            percentages,
            interpolation='nearest')

        selected = []
        cursor = 0
        for q, p in zip(quantiles, percentages):
            proc = True

            while proc:
                if cursor >= len(weights):
                    break

                w = weights[cursor]

                if q == w[1] or cursor + 1 == len(weights):
                    selected.append(w[0])
                    proc = False

                cursor += 1

        return selected

    @property
    def missing(self):
        return np.count_nonzero(np.isnan(self.trajectories))

    def ready(self):
        """Check that we have received all the values required to promote the next round"""
        return self.missing == 0

    def new_trials(self, trials):
        epoch = self.budget[self.round]['epoch']

        for i, trial in enumerate(trials):
            trial.params[self.fidelity.name] = epoch
            uid = trial.params[self.identity]

            self.trial2index[uid] = i
            self.index2trial[i] = uid
            
    def observe(self, params, result):
        super(DiversifiedSearch, self).observe(params, result)

        uid = params[self.identity]
        idx = self.trial2index[uid]
        epoch = params[self.fidelity.name]

        try:
            self.trajectories[epoch - 1, idx] = result
        except IndexError:
            print(f'Tried to insert {epoch - 1}, {idx} but shape: {self.trajectories.shape}')
            raise

    def is_done(self):
        return self.round + 1 == len(self.budget)

    def result(self):
        return sorted([self.trials[k] for k, _ in self.trial2index.items()], key=lambda t: t.objective)


builders = {
    'diversified': DiversifiedSearch,
}


def check():
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


if __name__ == '__main__':
    check()
