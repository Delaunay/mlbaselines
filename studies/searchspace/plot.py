import numpy

import matplotlib.pyplot as plt

from sspace import Space as SSpace

from scipy.optimize import OptimizeResult

try:
    from skopt import Space
    from skopt.plots import plot_objective, plot_evaluations
    from skopt.space import Real
    from skopt.utils import cook_estimator

    import_error = None
except ImportError as e:
    import_error = e


def orion_space_to_skopt_space(orion_space):
    """Convert Oríon's definition of problem's domain to a skopt compatible."""
    if import_error:
        raise import_error

    dimensions = []
    for key, dimension in orion_space.items():
        #  low = dimension._args[0]
        #  high = low + dimension._args[1]
        low, high = dimension.interval()
        # NOTE: A hack, because orion priors have non-inclusive higher bound
        #       while scikit-optimizer have inclusive ones.
        # pylint: disable = assignment-from-no-return
        high = numpy.nextafter(high, high - 1)
        shape = dimension.shape
        assert not shape or len(shape) == 1
        if not shape:
            shape = (1,)
        if dimension.prior_name == 'reciprocal':
            low = numpy.log(low)
            high = numpy.log(high)
        # Unpack dimension
        for i in range(shape[0]):
            dimensions.append(Real(name=key + '_' + str(i),
                                   prior='uniform',
                                   low=low, high=high))

    return Space(dimensions)


def xarray_to_points(space, objective, data):
    points = numpy.zeros((len(data.order), len(space.keys())))
    max_epochs = max(data.epoch.values)
    for i, (name, dim) in enumerate(space.items()):
        points[:, i] = data.loc[dict(epoch=max_epochs, seed=0)][name].values
        if dim.prior_name == 'reciprocal':
            points[:, i] = numpy.log(points[:, i])

    results = data[objective].loc[dict(epoch=max_epochs, seed=0)].values

    return points, results


def xarray_to_scipy_results(space, objective, data, model_seed=1):

    space = SSpace.from_dict(space)
    space = space.instantiate('Orion')
    X, y = xarray_to_points(space, objective, data)

    space = orion_space_to_skopt_space(space)

    model = cook_estimator('RF', space, random_state=model_seed)
    model.fit(X, y)

    best_idx = numpy.argmin(y)
    results = OptimizeResult()
    results.x = X[best_idx]
    results.fun = y[best_idx]
    results.models = [model]
    results.x_iters = X.tolist()
    results.func_vals = y.tolist()
    results.space = space
    results.specs = {}  # Maybe we don't need this one.
    return results


def plot(space, objective, data, filename, model_seed=1):

    results = xarray_to_scipy_results(space, objective, data, model_seed=1)

    axs = plot_objective(results)
    # axs = plot_evaluations(results)

    plt.savefig(filename, dpi=300)
