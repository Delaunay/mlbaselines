from typing import Union, Dict
from collections import OrderedDict
from sspace import Space

from olympus.hpo.fidelity import Fidelity
from olympus.utils import new_seed, warning, compress_dict, decompress_dict


class Trial:
    def __init__(self, params_instance):
        self.params = params_instance
        self.objectives = []

    @property
    def uid(self):
        return self.params['uid']

    def __repr__(self):
        return f'<Trial(params={self.params}, objective={self.objective})>'

    def observe(self, val):
        self.objectives.append(val)

    @property
    def objective(self):
        if self.objectives:
            return self.objectives[-1]
        return None

    def state_dict(self):
        return {
            'params': self.params,
            'objectives': self.objectives
        }

    @staticmethod
    def from_dict(state):
        t = Trial(state['params'])
        t.objectives = state['objectives']
        return t

    def __eq__(self, other):
        return self.params == other.params and self.objectives == other.objectives


class UnknownTrial(Exception):
    pass


class TrialDoesNotExist(Exception):
    pass


class WaitingForTrials(Exception):
    pass


class OptimizationIsDone(Exception):
    pass


class LogicError(Exception):
    pass


class HyperParameterOptimizer:
    """Search for optimal hyper parameter configuration

    Parameters
    ----------
    fidelity: Fidelity
        Set of values representing how confident we are on the objective.
        In machine learning it is often epoch

    params: Union[Space, Dict]
        A definition of the parameter space to search

    seed: int
        Seed for the pseudo random number generator
    """
    def __init__(self, fidelity: Fidelity, space: Union[Space, Dict], seed=new_seed(hpo_sampler=0), **kwargs):
        self.identity = 'uid'

        for k, v in kwargs.items():
            warning(f'used parameter ({k}: {v})')

        if isinstance(space, dict):
            space = Space.from_dict(space)

        if space is not None:
            space.identity(self.identity)

        if isinstance(fidelity, dict):
            fidelity = Fidelity.from_dict(fidelity)

        self.fidelity = fidelity
        self.space = space
        self.seed = seed
        self.seed_time = 0
        self.manual_insert = 0
        self.manual_samples = []
        self.manual_fidelity = []
        self.trials = OrderedDict()

    def insert_manual_sample(self, sample=None, fidelity_override=None, **kwargs):
        """Can be used to force a specific configuration to be considered

        Parameters
        ----------
        sample: dict
            A configuration of parameters to use

        fidelity_override: int
            The fidelity to set this particular trial, it enables hand picked configuration to bypass the HPO
            entirely

        kwargs:
            can be used to specify parameter override or the entire sample configuration

        Notes
        -----
        This function can also be used to run HPO `backtest` by inserting known samples and observing the result
        to check how the HPO should behave, or test a novel HPO.
        """
        if sample is None:
            sample = kwargs
        else:
            sample.update(kwargs)

        # do basic validation
        for k, v in self.space.space_tree.items():
            if k not in sample:
                raise KeyError(f'Manual sample should have (key: {k})')

        sample[self.identity] = self.space._compute_identity(sample)

        if fidelity_override is not None:
            self.manual_fidelity[sample[self.identity]] = fidelity_override

        self.manual_samples.append(sample)

    def _fetch_manual_samples(self, count):
        """Use manual samples first"""
        manual_slice = []

        # we have manual samples to use
        if len(self.manual_samples) > 0 and self.manual_insert < len(self.manual_samples):
            remaining = len(self.manual_samples) - self.manual_insert

            if count >= remaining:
                manual_slice = self.manual_samples[self.manual_insert:]
                self.manual_insert += remaining
            else:
                manual_slice = self.manual_samples[self.manual_insert:self.manual_insert+count]
                self.manual_insert += count

        return manual_slice

    def _apply_fidelity(self, manual_slice):
        """Apply fidelity override to manual samples if they have any"""
        if manual_slice is not None:
            for params in manual_slice:
                uid = params[self.identity]
                if uid in self.manual_fidelity:
                    params[self.fidelity.name] = self.manual_fidelity[uid]

    # @final
    def sample(self, count=1, **variables):
        """Sample new configurations and register them"""
        samples = []

        manual_slice = self._fetch_manual_samples(count)
        samples.extend(manual_slice)

        count -= len(manual_slice)

        if count > 0:
            new_samples = self.space.sample(count, seed=self.seed + self.seed_time, **variables)
            samples.extend(new_samples)

        self.seed_time += 1
        trials = []

        # Register all the samples
        for s in samples:
            t = Trial(s)
            trials.append(t)
            self.trials[s[self.identity]] = t

        self.new_trials(trials)
        self._apply_fidelity(manual_slice)

        return samples

    # @final
    def observe(self, identity: Union[str, OrderedDict], result):
        """Observe the result of a given trial

        identity: str or OrderedDict
            where str is the the identity string or OrderedDict is the full parameter dictionary

        Raises
        ------
        UnknownTrial
            if the trial cannot be found
        """
        uid = identity

        if isinstance(identity, (OrderedDict, dict)):
            uid = identity.get(self.identity)

        if uid is None:
            raise UnknownTrial(f'with (uid: {identity})')

        try:
            self.trials[uid].observe(result)
            self.new_result(uid, result)

        except KeyError as e:
            raise TrialDoesNotExist(f'with (uid: {identity})') from e

    def state_dict(self, compressed=False):
        state = {
            'fidelity': self.fidelity.to_dict(),
            'seed': self.seed,
            'seed_time': self.seed_time,
            'manual_insert': self.manual_insert,
            'space': self.space.serialize(),
            'manual_samples': self.manual_samples,
            'manual_fidelity': self.manual_fidelity,
            'trials': [
                (k, trial.state_dict()) for k, trial in self.trials.items()
            ]
        }

        if compressed:
            state = compress_dict(state)

        return state

    def load_state_dict(self, state):
        state = decompress_dict(state)

        self.space = Space.from_dict(state['space'])
        self.seed = state['seed']
        self.manual_samples = state['manual_samples']
        self.manual_fidelity = state['manual_fidelity']
        self.manual_insert = state['manual_insert']
        self.seed_time = state['seed_time']
        self.fidelity = Fidelity.from_dict(state['fidelity'])
        self.trials = OrderedDict(
            (k, Trial.from_dict(t)) for k, t in state['trials']
        )
        return self

    @staticmethod
    def from_dict(state):
        hpo = HyperParameterOptimizer(None, None)
        return hpo.load_state_dict(state)

    def new_trials(self, trials):
        """Event sent when a new configurations are sampled"""
        for trial in trials:
            trial.params[self.fidelity.name] = self.fidelity.max

    def new_result(self, identity, result):
        """Event sent when a new result is received"""
        pass

    def suggest(self, **variables):
        """Return configuration to run"""
        raise NotImplementedError()

    def is_done(self):
        """Return true if the optimization is finished"""
        raise NotImplementedError()

    def remaining(self):
        """Return the number of remaining trials, this is used to kill superfluous worker if possible"""
        return None

    def result(self):
        """Return the configuration with the smallest objective

        Returns
        -------
        result: Tuple[Dict, float]
            tuple with the parameters and the value of the objective
        """
        min = float('+inf')
        trial = None

        for k, t in self.trials.items():
            if t.objective < min:
                min = t.objective
                trial = t

        params = trial.params
        objective = trial.objective

        return params, objective

    def info(self):
        """Return information about the algo configuration

        Returns
        -------
        Dictionary, the content is dependant on the algorithm

        unique_samples: int
            Number of unique configuration that was sampled

        total_epochs: int
            Total number of epochs performed

        parallelism: int
            Average number of trial alive at the same time
        """
        raise NotImplementedError()

    class _ConfigurationIterator:
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.batch = None
            self.position = 0

        def __next__(self):
            if self.batch is None:
                try:
                    self.batch = self.optimizer.suggest()

                except WaitingForTrials:
                    raise

                except OptimizationIsDone:
                    raise StopIteration

            result = self.batch[self.position]
            self.position += 1

            if self.position >= len(self.batch):
                self.batch = None
                self.position = 0

            return result

    def __iter__(self):
        return self._ConfigurationIterator(self)

