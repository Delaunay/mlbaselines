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




def bootstrap(x, xv, xt, bootstrap_seed, hpo_done):
    ### TODO: add the two external test sets
    num_points = x.shape[0]
    n_train = int(num_points * 0.8)
    n_valid = int(num_points * 0.20)
    n_test = int(num_points * 0.10)
    indices = set(range(num_points))

    rng = np.random.RandomState(bootstrap_seed)

    train_set = sorted(rng.choice(list(indices), size=n_train, replace=True))

    indices -= set(train_set)

    valid_set = sorted(rng.choice(list(indices), size=n_valid, replace=True))

    indices -= set(valid_set)

    test_set = sorted(rng.random.choice(list(indices), size=n_test, replace=True))

    indices -= set(test_set)

    if hpo_done:
        train_set = train_set + valid_set
        test_set = test_set
    else:
        train_set = train_set
        test_set = valid_set

    return x[train_set], x[test_set]


def get_pcc(preds, targets):
    return np.corrcoef(preds, targets)[0,1]


def get_roc_auc(preds, targets):
	fpr, tpr, _  = roc_curve(targets, preds)
	auc_result = auc(fpr,tpr)

	return auc_result


def get_space():
    return {'some_hp': 'uniform(1, 10)',
	        'some_other_hp': 'loguniform(1, 10)'}


def main(bootstrap_seed, model_seed, hidden_layer_sizes, solver, alpha, ensembling = False, hpo_done=False):
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
    ensembling: bool
        decides if yes or no we will use ensembling for the test set
    hpo_done: bool
        If hpo_done is True, we train on train+valid and report on test. If hpo_done is False, we
        train on train, report on valid and ignore test.

    """

    #Create train/test spits using seed
    #dataset in matrix format with last column being the target'
    x = get_panallele_dataset(folder='NetMHC')
    xv = get_valid_dataset(folder='NetMHC')
    xt = get_test_dataset(folder='NetMHC')
    train, test = bootstrap(x, xv, xt, bootstrap_seed, hpo_done)

    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, solver=solver, alpha=alpha, random_state = model_seed)
    model.fit(train[:, :-1], train[:, -1])

    y_pred = model.predict(test[:, :-1])
    roc_auc = get_roc_auc(y_pred, test[:, -1])

    return {"objective": roc_auc}
