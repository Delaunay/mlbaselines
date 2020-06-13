"""
Convergence figures
Plot the progression of the HPO algorithms
"""
import copy
import os
import time

import xarray
import json
import gzip

import numpy
import pandas

import seaborn as sns
import matplotlib.pyplot as plt

import joblib

# Use joblib to speed things up when rerunning
mem = joblib.Memory('joblib_cache')

@mem.cache
def load_hpo_results(namespace, save_dir):
    with open(f'{save_dir}/hpo_{namespace}.json', 'r') as f:
        data = {hpo: xarray.Dataset.from_dict(d) for hpo, d in json.loads(f.read()).items()}

    return data



@mem.cache
def load_variance_results(namespace, save_dir):
    with open(f'{save_dir}/variance_{namespace}.json', 'r') as f:
        data = xarray.Dataset.from_dict(json.loads(f.read()))

    return data


# std = 'mean-normalized'
std = 'raw'

# reporting_set = 'valid'
reporting_set = 'test'


ROOT = '/home/bouthilx/Dropbox/Olympus-Data'
VAR_ROOT = os.path.join(ROOT, 'variance')
HPO_ROOT = os.path.join(ROOT, 'hpo')


case_studies = {
    'vgg': 'vgg',
    'segmentation': 'segmentation',
    'bert-sst2': 'sst2',
    'bert-rte': 'rte',
    'bio-task2': 'bio-task2'}
    # 'logreg': 'logreg'}


objectives = {
    'vgg': 'test_error_rate',
    'segmentation': 'test_mean_jaccard_distance',
    'bert-sst2': 'test_error_rate',
    'bert-rte': 'test_error_rate',
    'bio-task2': 'test_aac',
    'logreg': 'test_error_rate'}


data = {}


start = time.time()
for key, name in case_studies.items():
    print(f'Loading {key}')
    data[key] = {
        # 'variance': load_variance_results(name + '-var', VAR_ROOT),
        'hpo': load_hpo_results(name + '-hpo', HPO_ROOT)
    }

elapsed_time = time.time() - start
print(f'It took {elapsed_time}s to load all data.')

# Gather data

def cum_argmin(x):
    """ Return the indices corresponding to an cumulative minimum
        (numpy.minimum.accumulate)
    """
    minima = numpy.minimum.accumulate(x, axis=0)
    diff = numpy.diff(minima, axis=0)
    jumps = numpy.vstack(numpy.arange(x.shape[0]) for _ in range(x.shape[1])).T
    jumps[1:, :] *= (diff != 0)
    jumps = numpy.maximum.accumulate(jumps, axis=0)
    return jumps


def cum_argmin(x):
    """ Return the indices corresponding to an cumulative minimum
        (numpy.minimum.accumulate)
    """
    minima = numpy.minimum.accumulate(x)
    diff = numpy.diff(minima)
    jumps = numpy.arange(len(x))
    jumps[1:] *= (diff != 0)
    jumps = numpy.maximum.accumulate(jumps)
    return jumps


# TODO: finish this here to compute correctly the error regrets
def test_cum(x, valid, test):
    regrets_idx = cum_argmin(x[valid])
    x[valid] = x[valid].to_numpy()[regrets_idx]
    x[test] = x[test].to_numpy()[regrets_idx]
    return x


fig = plt.figure(figsize=(7, 3.5))
axes = fig.subplots(2, len(data),
                    gridspec_kw={'left': .04,
                        'top': .98, 'right': .99, 'bottom': .12,
                        'hspace': 0.02, 'wspace': 0.07}
                    )
axes = axes.T
sns.despine()


def compute_regrets(key, case_data):
    metrics = [objectives[key].replace('test', 'validation'), objectives[key]]

    n_seeds = case_data['hpo']['random_search']['seed'].shape[0]

    # Extract pandas dataframe to plot
    valid_df = list()
    test_df = list()
    for name, this_data in case_data['hpo'].items():
        if name in ('grid_search', 'nudged_grid_search'):
            continue

        # Only take smallest budget all bayesopts achieved
        if name == 'bayesopt':
            seeds = list(this_data[metrics[0]].dropna(dim='seed', how='all').seed.values)
            print(len(seeds))
            this_data = this_data.sel(seed=seeds)
            thresh = min(len(seeds), 10)
            order = list(this_data[metrics[0]].dropna(dim='order', how='all', thresh=thresh).order.values)
            print(order)
            if not order:
                continue
            this_data = this_data.sel(order=order)

        # Here we don't need the noise and params
        this_data = this_data.drop('noise').drop('params')

        this_data = this_data.to_dataframe()
        to_plot = this_data[metrics]
        to_plot = to_plot.reset_index()
        # No early stopping: take the last epoch
        to_plot = to_plot.query('epoch == epoch.max()')
        to_plot = to_plot.drop('epoch', axis=1)

        if name in ('grid_search', 'nudged_grid_search'):
            # Create pseudo replicates of the experiment by randomizing the
            # order in which we cross the grid
            pseudo_exps = list()
            for i in range(n_seeds):
                this_data = to_plot.copy()
                this_data['seed'] = i
                this_error_rate = to_plot[metrics].values.copy()
                numpy.random.shuffle(this_error_rate)
                this_data[metrics] = this_error_rate
                pseudo_exps.append(this_data)
            to_plot = pandas.concat(pseudo_exps)

        regrets = to_plot.groupby('seed')[metrics].apply(
            test_cum, valid=metrics[0], test=metrics[1])

        valid_to_plot = copy.deepcopy(to_plot)
        valid_to_plot['regret'] = regrets[metrics[0]]
        valid_to_plot = valid_to_plot.drop('seed', axis=1)
        valid_to_plot['hpo'] = name
        valid_df.append(valid_to_plot)

        test_to_plot = copy.deepcopy(to_plot)
        test_to_plot['regret'] = regrets[metrics[1]]
        test_to_plot = test_to_plot.drop('seed', axis=1)
        test_to_plot['hpo'] = name
        test_df.append(test_to_plot)

    valid_df = pandas.concat(valid_df, axis=0)
    test_df = pandas.concat(test_df, axis=0)
    return valid_df, test_df


start = time.time()
for ax, (key, case_data) in zip(axes, data.items()):
    # valid_df, test_df = mem.cache(compute_regrets)(key, case_data)
    valid_df, test_df = compute_regrets(key, case_data)
    valid_df = valid_df.dropna()
    test_df = test_df.dropna()
    print(key)
    print(valid_df)
    print(test_df)

    sns.lineplot(data=valid_df, x='order', y='regret', hue='hpo', ax=ax[0],
                 legend=False)
    sns.lineplot(data=test_df, x='order', y='regret', hue='hpo', ax=ax[1],
                 legend='brief' if key == 'bert-rte' else False)
    if key == 'bert-rte':
        l = ax[1].legend()
        # Remove the first patch, which is an empty marker added by
        # seaborn
        ax[1].legend(l.legendHandles[1:],
                     [s.get_label().replace('_', ' ')
                      for s in l.legendHandles[1:]],
                      loc=(-.05, .05))


    ax[0].text(.9, .9, key, ha='right', transform=ax[0].transAxes)

    ax[0].set_facecolor('.95')
    ax[1].set_facecolor('.95')

    # vgg
    if key == 'vgg':
        ax[0].set_ylim(0.075, 0.125)
        ax[1].set_ylim(0.075, 0.125)
    elif key == 'segmentation':
        ax[0].set_ylim(0.44, 0.49)
        ax[1].set_ylim(0.44, 0.49)
    elif key == 'bert-sst2':
        ax[0].set_ylim(0.035, 0.05)
        ax[1].set_ylim(0.035, 0.05)
    elif key == 'bert-rte':
        ax[0].set_ylim(0.24, 0.35)
        ax[1].set_ylim(0.24, 0.35)
    # else:
    #     raise ValueError(key)
    ax[1].set_xlabel('HPO iterations')
    if ax[0] is axes[0, 0]:
        ax[0].set_ylabel('Validation error')
        ax[1].set_ylabel('Test error')
        # sns.despine(ax=ax[0], bottom=True, left=False)
        sns.despine(ax=ax[0], bottom=True, left=False)
        ax[0].get_xaxis().set_visible(False)
        ax[0].set_yticks(())
        ax[1].set_yticks(())
    else:
        sns.despine(ax=ax[0], bottom=True, left=True)
        sns.despine(ax=ax[1], bottom=False, left=True)
        # sns.despine(ax=ax[1], bottom=True, left=True)
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)

# plt.tight_layout()
plt.savefig('test_many_hpo.png', dpi=300)
plt.savefig('test_many_hpo.pdf')
