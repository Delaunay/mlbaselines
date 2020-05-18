import time
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
import pdb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor,MLPClassifier
import bio_datasets
import bio_metrics




def bootstrap(x, bootstrap_seed, hpo_done):

    num_points = x.shape[0]
    n_train = int(num_points * 0.8)
    n_valid = int(num_points * 0.10)
    n_test = int(num_points * 0.10)
    indices = set(range(num_points))

    rng = np.random.RandomState(bootstrap_seed)

    train_set = sorted(rng.choice(list(indices), size=n_train, replace=True))

    indices -= set(train_set)

    valid_set = sorted(rng.choice(list(indices), size=n_valid, replace=True))

    indices -= set(valid_set)

    test_set = sorted(rng.random.choice(list(indices), size=n_test, replace=True))

    indices -= set(test_set)

    #code below is redundant, just for clarity / explicitness
    if hpo_done:
        train_set = train_set + valid_set
        test_set = test_set
    else:
        train_set = train_set
        test_set = valid_set

    return x[train_set], x[test_set]

def get_space():
    return {'some_hp': 'uniform(1, 10)',
	        'some_other_hp': 'loguniform(1, 10)'}


def main(bootstrap_seed, model_seed, hidden_layer_sizes, solver, alpha, hpo_done=False):
    """

    Parameters
    ----------
    bootstrap_seed: int
        seed for controling which data-points are selected for training/testing splits
    model_seed: int
        seed for the generation of weights
    hidden_layer_sizes: tuple
        the size of layers ex: (50,) is one layer of 50 neurons
    solver: one of {‘lbfgs’, ‘sgd’, ‘adam’}
        solver to use for optimisation
    alpha: float
        L2 penalty (regularization term) parameter.
    hpo_done: bool
        If hpo_done is True, we train on train+valid and report on test. If hpo_done is False, we
        train on train, report on valid and ignore test.

    """


    x = get_singleallele_dataset(allele='HLA-A02:01', folder='NetMHC')
    train, test = bootstrap(x, bootstrap_seed, hpo_done)

    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, solver=solver, alpha=alpha)
    model.fit(train[:, :-1], train[:, -1])

    y_pred = model.predict(test[:, :-1])
    roc_auc = get_roc_auc(y_pred, test[:, -1])

    return {"objective": roc_auc}
