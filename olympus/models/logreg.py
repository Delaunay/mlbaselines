import numpy

from olympus.models.mlp import MLP


def logistic_regression(input_size, output_size):
    """Logistic regression is a statistical model that in its basic form uses a logistic function to
    model a binary dependent variable, although many more complex extensions exist.
    In regression analysis, logistic regression (or logit regression) is
    estimating the parameters of a logistic model (a form of binary regression).
    More on `wikipedia <https://en.wikipedia.org/wiki/Logistic_regression>`_.

    See also :class`.MLP`

    """

    if not isinstance(input_size, int):
        input_size = numpy.product(input_size)

    if not isinstance(output_size, int):
        output_size = numpy.product(output_size)

    return MLP(input_size, output_size, layers=[], bias=True)


builders = {'logreg': logistic_regression}
