import_error = None
try:
    import pandas as pd
    import numpy as np

    from fanova import fANOVA
    from fanova.visualizer import Visualizer

    from sspace.space import Space
    from olympus.utils.functional import select
except ImportError as e:
    import_error = e


class FANOVA:
    """
    Parameters
    ----------
    data: DataFrame

    hp_names: List[str]
        list of hyper parameter names (to extract from the DataFrame)

    objective: str
        name of the objective

    References
    ----------
    .. [1] F. Hutter and H. Hoos and K. Leyton-Brown
        "An Efficient Approach for Assessing Hyperparameter Importance"
        Proceedings of International Conference on Machine Learning 2014 (ICML 2014).
    """

    def __init__(self, data: pd.DataFrame, hp_names, objective, hp_space=None):
        if import_error is not None:
            raise import_error

        x = data[hp_names]
        y = data[objective]

        self.space = hp_space
        self.hp_names = hp_names
        self.fanova = fANOVA(x.values, y.values)
        self.size = len(hp_names)
        self._importance = np.zeros((self.size, self.size))
        self._importance_std = np.zeros_like(self._importance)
        self.vis = Visualizer(self.fanova, Space.from_dict(hp_space).instantiate(), '/tmp', 'objective')

        self._compute_importance()

    def compute_marginal(self, index, marginals=None):
        """Compute the effect a change on the hyper parameter value has on the objective value

        Returns
        -------
        returns a list of dictionaries (name, objective, value, std)
        """
        data = self.vis.generate_marginal(index)
        marginals = select(marginals, [])

        # (mean, std, grid)
        if len(data) > 2:
            for y, std, x in zip(data[0], data[1], data[2]):
                marginals.append(dict(
                    name=self.hp_names[index],
                    objective=y,
                    value=x,
                    std=std))
        # (mean, std)
        else:   # Categorical
            choices = self._get_choices(index)

            for (y, std), x in zip(data, choices):
                marginals.append(dict(
                    name=self.hp_names[index],
                    objective=y,
                    value=x,
                    std=std))

        return marginals

    def _get_choices(self, param):
        """For categorical HP retrieve the choices available"""
        from ConfigSpace import CategoricalHyperparameter
        p, p_name, p_idx = self.vis._get_parameter(param)

        if isinstance(p, CategoricalHyperparameter):
            return p.choices

        return ['Constant']

    def compute_marginals(self):
        marginals = []

        for i, _ in enumerate(self.hp_names):
            self.compute_marginal(i, marginals)

        return marginals

    @property
    def importance(self):
        """TODO: doc

        Returns
        -------
        Importance matrix of pairs of hyper parameters
        """
        return self._importance

    @property
    def importance_std(self):
        """TODO: doc

        Returns
        -------
        Standard deviation of the importance matrix of pairs of hyper parameters
        """
        return self._importance_std

    def _compute_importance(self):
        importance = self.fanova.quantify_importance(list(range(self.size )))
        self.importance_long = []

        for k, values in importance.items():
            if len(k) == 1:
                i = k[0]
                j = k[0]
            elif len(k) == 2:
                i = k[0]
                j = k[1]
            else:
                continue

            self._importance[i, j] = values['total importance']
            self._importance_std[i, j] = values['total std']
            self.importance_long.append(dict(
                row=self.hp_names[i],
                col=self.hp_names[j],
                importance=values['total importance'],
                std=values['total std']
            ))
