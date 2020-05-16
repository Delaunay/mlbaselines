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

    if hpo_done:
        train_set = train_set + valid_set
        test_set = test_set
    else:
        train_set = train_set
        test_set = valid_set

    return x[train_set], x[test_set]


def get_roc_auc(preds, targets):
	fpr, tpr, _  = roc_curve(targets, preds)
	auc_result = auc(fpr,tpr)

	return auc_result

def get_pcc(preds, targets):
	pcc = np.corrcoef(preds, targets)[0,1]
	return pcc


def get_space():
    return {'some_hp': 'uniform(1, 10)',
	        'some_other_hp': 'loguniform(1, 10)'}


def main(bootstrap_seed, model_seed, some_hp, some_other_hp, hpo_done=False):
    """

    Parameters
    ----------
    bootstrap_seed: int
        seed for controling which data-points are selected for training/testing splits
    algo_seed: int
        seed for the algorithm
    some_hp: mystery
        some hyperparameter to set...
    some_other_hp: mystery
        some other hyperparameter to set...
    hpo_done: bool
        If hpo_done is True, we train on train+valid and report on test. If hpo_done is False, we
        train on train, report on valid and ignore test.

    """

    #Create train/test spits using seed
    #'some dataset in matrix format with last column being the target'
    x = get_panallele_dataset(folder='NetMHC')
    train, test = bootstrap(x, bootstrap_seed, hpo_done)

    model = MLPRegressor(hidden_layer_sizes=(2,), solver="sgd", alpha=0)
    model.fit(train[:, :-1], train[:, -1])

    y_pred = model.predict(test[:, :-1])
    error_rate = get_pcc(y_pred, test[:, -1])

    return {"objective": error_rate}
