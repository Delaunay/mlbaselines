import numpy

import george


from robo.priors.default_priors import DefaultPrior
from robo.models.wrapper_bohamiann import WrapperBohamiann
from robo.models.gaussian_process import GaussianProcess
from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.models.random_forest import RandomForest
from robo.maximizers.scipy_optimizer import SciPyOptimizer
from robo.maximizers.random_sampling import RandomSampling
from robo.maximizers.differential_evolution import DifferentialEvolution
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.acquisition_functions.ei import EI
from robo.acquisition_functions.pi import PI
from robo.acquisition_functions.log_ei import LogEI
from robo.acquisition_functions.lcb import LCB
from robo.acquisition_functions.marginalization import MarginalizationGPMCMC
from robo.initial_design import init_latin_hypercube_sampling

from sspace import Space
from sspace.space import compute_identity

from olympus.hpo.optimizer import Trial, HyperParameterOptimizer, WaitingForTrials, OptimizationIsDone
from olympus.hpo.fidelity import Fidelity
from olympus.utils import new_seed, compress_dict, decompress_dict
from olympus.utils.functional import unflatten, encode_rng_state, decode_rng_state


def build_model(lower, upper, model_type="gp_mcmc", model_seed=1, prior_seed=1):
    """
    General interface for Bayesian optimization for global black box
    optimization problems.

    Parameters
    ----------
    lower: numpy.ndarray (D,)
        The lower bound of the search space
    upper: numpy.ndarray (D,)
        The upper bound of the search space
    model_type: {"gp", "gp_mcmc", "rf", "bohamiann", "dngo"}
        The model for the objective function.
    model_seed: int
        Seed for random number generator of the model 
    prior_seed: int
        Seed for random number generator of the prior

    Returns
    -------
        Model
    """
    assert upper.shape[0] == lower.shape[0], "Dimension miss match"
    assert numpy.all(lower < upper), "Lower bound >= upper bound"

    cov_amp = 2
    n_dims = lower.shape[0]

    initial_ls = numpy.ones([n_dims])
    exp_kernel = george.kernels.Matern52Kernel(initial_ls,
                                               ndim=n_dims)
    kernel = cov_amp * exp_kernel

    prior = DefaultPrior(len(kernel) + 1, numpy.random.RandomState(prior_seed))

    n_hypers = 3 * len(kernel)
    if n_hypers % 2 == 1:
        n_hypers += 1

    # NOTE: Some models do not support RNG properly and rely on global RNG state
    #       so we need to seed here as well...
    numpy.random.seed(model_seed)
    model_rng = numpy.random.RandomState(model_seed)
    if model_type == "gp":
        model = GaussianProcess(kernel, prior=prior, rng=model_rng,
                                normalize_output=False, normalize_input=True,
                                lower=lower, upper=upper)
    elif model_type == "gp_mcmc":
        model = GaussianProcessMCMC(kernel, prior=prior,
                                    n_hypers=n_hypers,
                                    chain_length=200,
                                    burnin_steps=100,
                                    normalize_input=True,
                                    normalize_output=False,
                                    rng=model_rng, lower=lower, upper=upper)

    elif model_type == "rf":
        model = RandomForest(rng=model_rng)

    elif model_type == "bohamiann":
        model = WrapperBohamiann()

    elif model_type == "dngo":
        from pybnn.dngo import DNGO

        model = DNGO()

    else:
        raise ValueError("'{}' is not a valid model".format(model_type))

    return model


def build_optimizer(model, maximizer="random", acquisition_func="log_ei", maximizer_seed=1):
    """
    General interface for Bayesian optimization for global black box
    optimization problems.

    Parameters
    ----------
    maximizer: {"random", "scipy", "differential_evolution"}
        The optimizer for the acquisition function.
    acquisition_func: {"ei", "log_ei", "lcb", "pi"}
        The acquisition function
    maximizer_seed: int
        Seed for random number generator of the acquisition function maximizer

    Returns
    -------
        Optimizer
    """

    if acquisition_func == "ei":
        a = EI(model)
    elif acquisition_func == "log_ei":
        a = LogEI(model)
    elif acquisition_func == "pi":
        a = PI(model)
    elif acquisition_func == "lcb":
        a = LCB(model)
    else:
        raise ValueError("'{}' is not a valid acquisition function"
                         .format(acquisition_func))

    if isinstance(model, GaussianProcessMCMC):
        acquisition_func = MarginalizationGPMCMC(a)
    else:
        acquisition_func = a

    maximizer_rng = numpy.random.RandomState(maximizer_seed)
    if maximizer == "random":
        max_func = RandomSampling(acquisition_func, model.lower, model.upper, rng=maximizer_rng)
    elif maximizer == "scipy":
        max_func = SciPyOptimizer(acquisition_func, model.lower, model.upper, rng=maximizer_rng)
    elif maximizer == "differential_evolution":
        max_func = DifferentialEvolution(acquisition_func, model.lower, model.upper,
                                         rng=maximizer_rng)
    else:
        raise ValueError("'{}' is not a valid function to maximize the "
                         "acquisition function".format(maximizer))

    # NOTE: Internal RNG of BO won't be used.
    # NOTE: Nb of initial points won't be used within BO, but rather outside
    bo = BayesianOptimization(lambda: None, model.lower, model.upper,
                              acquisition_func, model, max_func,
                              initial_points=None, rng=None,
                              initial_design=init_latin_hypercube_sampling,
                              output_path=None)

    return bo


def build_bounds(space):
    lower = numpy.zeros(len(space.keys()))
    upper = numpy.zeros(len(space.keys()))
    for i, (name, dim) in enumerate(space.items()):
        lower[i], upper[i] = dim.interval()
        if dim.prior_name == 'reciprocal':
            lower[i] = numpy.log(lower[i])
            upper[i] = numpy.log(upper[i])

    return lower, upper


class RoBO(HyperParameterOptimizer):
    """
    Wrapper for RoBO


    .. TODO: add citations and more details

    Parameters
    ----------
    model_type: {"gp", "gp_mcmc", "rf", "bohamiann", "dngo"}
        The model for the objective function.
    maximizer: {"random", "scipy", "differential_evolution"}
        The optimizer for the acquisition function.
    acquisition_func: {"ei", "log_ei", "lcb", "pi"}
        The acquisition function
    n_init: int
        Number of points for the initial design.
    model_seed: int
        Seed for random number generator of the model 
    prior_seed: int
        Seed for random number generator of the prior
    init_seed: int
        Seed for random number generator of the initial design
    maximizer_seed: int
        Seed for random number generator of the acquisition function maximizer

    """

    def __init__(self, fidelity: Fidelity, space: Space, count: int,
                 model_type='gp_mcmc',
                 maximizer="random", 
                 acquisition_func="log_ei",
                 n_init=10,
                 model_seed=new_seed(hpo_sampler=0),
                 prior_seed=new_seed(hpo_prior=0),
                 init_seed=new_seed(hpo_init=0),
                 maximizer_seed=new_seed(hpo_maximizer=0),
                 **kwargs):
        super(RoBO, self).__init__(fidelity, space, model_seed, **kwargs)
        self.count = count
        self.orion_space = self.space.instantiate('Orion')
        self.init_seed = init_seed
        self.n_init = n_init
        self.model_type = model_type

        assert n_init <= count, \
            "Number of initial design point has to be <= than the number of trials"

        lower, upper = build_bounds(self.orion_space)

        model = build_model(lower, upper, model_type, model_seed, prior_seed)
        self.robo = build_optimizer(
            model, maximizer=maximizer, acquisition_func=acquisition_func,
            maximizer_seed=maximizer_seed)

        # TODO: Need to test everything.... But so far seems complete. :)

    def sample(self, count=1, **variables):

        assert count <= self.n_init

        init = self.robo.initial_design(
            self.robo.lower,
            self.robo.upper,
            self.n_init,
            rng=numpy.random.RandomState(self.init_seed))

        samples = []
        trials = []
        for point in init[-count:]:
            sample, trial = self.create_and_register_new_point(point, **variables)
            samples.append(sample)
            trials.append(trial)

        self.new_trials(trials)

        return samples

    def is_waiting(self):
        for trial in self.trials.values():
            if not trial.objectives:
                return True

    @property
    def X(self):
        X = numpy.zeros((len(self.trials), len(self.orion_space.keys())))
        for i, trial in enumerate(self.trials.values()):
            for j, (name, dim) in enumerate(self.orion_space.items()):
                X[i, j] = trial.params[name]
                if dim.prior_name == 'reciprocal':
                    X[i, j] = numpy.log(trial.params[name])

        return X

    @property
    def y(self):
        y = numpy.zeros(len(self.trials))
        for i, trial in enumerate(self.trials.values()):
            y[i] = trial.objectives[-1]

        return y

    def choose_next(self, **variables):
        # Choose next point to evaluate
        point = self.robo.choose_next(self.X, self.y)

        sample, trial = self.create_and_register_new_point(point, **variables)

        self.new_trials([trial])

        return sample

    def create_and_register_new_point(self, point, **variables):
        sample = dict()
        for i, (name, dim) in enumerate(self.orion_space.items()):
            sample[name] = point[i]
            if dim.prior_name == 'reciprocal':
                sample[name] = numpy.exp(sample[name])

        sample.update(variables)
        sample = unflatten(sample)
        sample[self.identity] = compute_identity(sample, self.space._identity_size)

        trial = Trial(sample)
        self.trials[sample[self.identity]] = trial

        return sample, trial

    def suggest(self, **variables):
        if len(self.trials) < self.n_init:
            return self.sample(self.n_init- len(self.trials), **variables)

        if self.is_done():
            raise OptimizationIsDone()

        if self.count_done() < len(self.trials):
            raise WaitingForTrials()

        return [self.choose_next(**variables)]

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
            'parallelism': 1
        }

    def load_state_dict(self, state):
        state = decompress_dict(state)

        super(RoBO, self).load_state_dict(state)

        self.count = state['count']
        state['count'] = self.count
        numpy.random.set_state(decode_rng_state(state['global_oh_my_god_numpy_rng_state']))
        self.robo.maximize_func.rng.set_state(decode_rng_state(state['maximizer_rng_state']))
        model = self.robo.model
        model.rng.set_state(decode_rng_state(state['model_rng_state']))
        model.prior.rng.set_state(decode_rng_state(state['prior_rng_state']))
        if self.model_type == 'gp_mcmc':
            if state.get('model_p0', None) is not None:
                model.p0 = numpy.array(state['model_p0'])
                model.burned = True
            elif hasattr(model, 'p0'):
                delattr(model, 'p0')
                model.burned = False
        else:
            model.kernel.set_parameter_vector(state['model_kernel_parameter_vector'])
            model.noise = state['noise']

    def state_dict(self, compressed=True):
        state = super(RoBO, self).state_dict()
        state = super(RoBO, self).state_dict(compressed=False)
        state['count'] = self.count
        state['global_oh_my_god_numpy_rng_state'] = encode_rng_state(numpy.random.get_state())
        state['maximizer_rng_state'] = encode_rng_state(self.robo.maximize_func.rng.get_state())
        model = self.robo.model
        state['model_rng_state'] = encode_rng_state(model.rng.get_state())
        state['prior_rng_state'] = encode_rng_state(model.prior.rng.get_state())
        if self.model_type == 'gp_mcmc':
            if hasattr(model, 'p0'):
                state['model_p0'] = model.p0.tolist()
        else:
            state['model_kernel_parameter_vector'] = model.kernel.get_parameter_vector().tolist()
            state['noise'] = model.noise

        if compressed:
            state = compress_dict(state)

        return state

    def remaining(self):
        return self.count - self.count_done()


builders = {
    'robo': RoBO
}
