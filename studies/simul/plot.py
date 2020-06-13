# Table
#       Ideal | HP-fixed | N-fixed | Sum    | Simul
# vgg | mean  | mean+-     mean+-    mean+-
#     | var   | var+-  
# rte | mean  | mean+-     mean+-    mean+-
#     | var   | var+-  


# Load simuls
# 

import json
import time

import numpy
import xarray

import seaborn as sns

from matplotlib import pyplot as plt

import joblib

# Use joblib to speed things up when rerunning
mem = joblib.Memory('joblib_cache')

@mem.cache
def load_simul_results(namespace, save_dir):
    with open(f'{save_dir}/simul_{namespace}.json', 'r') as f:
        data = {
            hpo: {
                rep_type: xarray.Dataset.from_dict(d)
                for rep_type, d in reps.items()
            }
            for hpo, reps in json.loads(f.read()).items()
        }

    return data

# std = 'mean-normalized'
std = 'raw'

# reporting_set = 'valid'
reporting_set = 'test'



VAR_ROOT = 'olympus/studies/variance/data'
HPO_ROOT = 'olympus/studies/hpo/data'
SIMUL_ROOT = 'olympus/studies/simul/data'


colors_strs = """\
#86aec3
#0f68a4
#a2cf7a
#23901c
#eb8a89
#d30a0c
#edaf5f
#ef6f00
#baa2c6
#5a2d8a
#efef89
#a16918""".split('\n')


colors = dict(zip([
    'Bootstrap',
    'Weights init',
    'Data order',
    'Dropout',
    'Data augment',
    'Unknown',
    'Random Search',
    'Noisy Grid Search',
    'Bayes Opt'], colors_strs))


DEFAULT_COLOR = '#000000'

LABELS = {
    'bootstrap_seed': 'Bootstrap',
    'bootstrapping_seed': 'Bootstrap',
    'random_state': 'Weights init',
    'init_seed': 'Weights init',
    'sampler_seed': 'Data order',
    'transform_seed': 'Data augment',
    'global_seed': 'Dropout',
    'reference': 'Unknown',
    'vgg': 'CIFAR10\nVGG11',
    'bert-sst2': 'Glue-SST2\nBERT',
    'bert-rte': 'Glue-RTE\nBERT',
    'bio-task2': 'MHC\nMLP',
    'segmentation': 'PascalVOC\nResNet',
    'logreg': 'Breast Cancer\nLog Reg',
    'random_search-25': 'Random Search - 25',
    'random_search-50': 'Random Search',
    'random_search-100': 'Random Search',
    'random_search-200': 'Random Search - 200',
    'noisy_grid_search-25': 'Noisy Grid Search - 25',
    'noisy_grid_search-50': 'Noisy Grid Search',
    'noisy_grid_search-100': 'Noisy Grid Search',
    'noisy_grid_search-200': 'Noisy Grid Search - 200',
    'bayesopt-25': 'Bayes Opt - 25',
    'bayesopt-50': 'Bayes Opt',
    'bayesopt-100': 'Bayes Opt',
    'bayesopt-200': 'Bayes Opt - 200'
    }


for key, label in LABELS.items():
    colors[key] = colors.get(label, DEFAULT_COLOR)


case_studies = {
    'vgg': 'vgg',
    'segmentation': 'segmentation',  # TODO: Set segmentation when data is ready
    'bert-sst2': 'sst2',
    'bert-rte': 'rte',
    'bio-task2': 'rte'}  # TODO: Set bio-task2 when data is ready
    # 'logreg': 'logreg'}


objectives = {
    'vgg': 'test_error_rate',
    'segmentation': 'test_mean_jaccard_distance',
    # 'segmentation': 'test_error_rate',
    'bert-sst2': 'test_error_rate',
    'bert-rte': 'test_error_rate',
    'bio-task2': 'test_error_rate'}
    # 'bio-task2': 'test_aac'}
    # 'logreg': 'test_error_rate'}

IGNORE = []  # 'bio-task2']


data = {}

start = time.clock()
for key, name in case_studies.items():
    print(f'Loading {key}')
    data[key] = load_simul_results(name + '-simul', SIMUL_ROOT)

elapsed_time = time.clock() - start
print(f'It took {elapsed_time}s to load all data.')

# Gather data

def cum_argmin(x):
    """ Return the indices corresponding to an cumulative minimum
        (np.minimum.accumulate)
    """
    minima = numpy.minimum.accumulate(x, axis=0)
    diff = numpy.diff(minima, axis=0)
    jumps = numpy.vstack(numpy.arange(x.shape[0]) for _ in range(x.shape[1])).T
    jumps[1:, :] *= (diff != 0)
    jumps = numpy.maximum.accumulate(jumps, axis=0)
    return jumps


def get_test_metrics(valid, test):
    regrets_idx = cum_argmin(valid)

    regrets_idx = numpy.minimum(
        regrets_idx, test.shape[0] * numpy.ones(regrets_idx.shape) - 1).astype(int)

    return test[regrets_idx[-1], numpy.arange(valid.shape[1])]


SIMUL_LABEL = 'Simulation'
HP_FIXED_LABEL = 'HP-fixed'
IDEAL_LABEL = 'Ideal'
BOOTSTRAP_LABEL = 'Bootstrap'
WEIGHTS_LABEL = 'Weighs init'
HP_P_SIMUL_LABEL = 'HP+simul'
rep_types = [SIMUL_LABEL, HP_P_SIMUL_LABEL, HP_FIXED_LABEL, BOOTSTRAP_LABEL, WEIGHTS_LABEL,
             IDEAL_LABEL]
# rep_types = [HP_FIXED_LABEL, BOOTSTRAP_LABEL, WEIGHTS_LABEL]

STD_OFFSET = 100
X_TICKS_FONT_SIZE = 8


NUM_REPLICATES = 100


start = time.clock()
stats = {}
for key, case_data in data.items():
    print(f'Computing stats for {key}')

    case_stats = {'means': dict(), 'stds': dict()}
    stats[key] = case_stats

    case_data = case_data['random_search']
    hpo_data = case_data['ideal']
    max_epoch = int(hpo_data.epoch.max())
    hpo_data = hpo_data.sel(epoch=max_epoch)

    valid = hpo_data[objectives[key].replace('test', 'validation')].values
    test = hpo_data[objectives[key]].values
    
    # NOTE: There is 50 HPOs of 200 points. 
    # We divide the 200 in sets of 100 and get the equivalent of 100 HPOs (random search)
    # We did this because we needed 200 points per HPO to fit the surrogate models but
    # we simulate for budgets of 100 trials.
    n_ideals = len(hpo_data.seed)
    print(n_ideals)
    ideal_data = numpy.ones(n_ideals * 2) * numpy.nan
    ideal_data[:n_ideals] = get_test_metrics(valid[:100], test[:100])
    ideal_data[n_ideals:] = get_test_metrics(valid[100:], test[100:])

    # case_stats['means'][IDEAL_LABEL] = numpy.atleast_1d(ideal_data[:NUM_REPLICATES].mean())
    case_stats['means'][IDEAL_LABEL] = ideal_data # [:20]
    case_stats['stds'][IDEAL_LABEL] = numpy.atleast_1d(ideal_data[:NUM_REPLICATES].std()) * STD_OFFSET
    case_stats['points'] = ideal_data

    order = range(min(NUM_REPLICATES, len(case_data['biased'].order)))
    hpo_data = case_data['biased'].sel(epoch=max_epoch, order=order)  # HP-fixed
    case_stats['means'][HP_FIXED_LABEL] = hpo_data[objectives[key]].mean(dim='order').values
    case_stats['stds'][HP_FIXED_LABEL] = hpo_data[objectives[key]].std(dim='order').values * STD_OFFSET

    print(float(hpo_data[objectives[key]].std(dim='order').values.std()))

    simul_fix_data = case_data['simul-fix'].sel(epoch=max_epoch)
    simul_fixed_means = simul_fix_data[objectives[key]].mean(dim='order')
    print(simul_fix_data[objectives[key]].shape)
    print(float(simul_fixed_means.mean()))
    simul_fixed_stds = simul_fix_data[objectives[key]].std(dim='order')
    print(float(simul_fixed_stds.mean()))
    print(float(simul_fixed_stds.std()))

    print('hp-fixed + simul-fixed')
    simul_fixed_stds = simul_fix_data[objectives[key]].std(dim='order').values
    hp_fixed_stds = case_stats['stds'][HP_FIXED_LABEL] / STD_OFFSET
    std_sum = numpy.sqrt(simul_fixed_stds**2 + hp_fixed_stds**2)
    print(std_sum.mean())
    print(std_sum.std())
    case_stats['means'][HP_P_SIMUL_LABEL] = case_stats['means'][HP_FIXED_LABEL]
    case_stats['stds'][HP_P_SIMUL_LABEL] = std_sum * STD_OFFSET

    order = range(min(NUM_REPLICATES, len(case_data['biased'].order)))
    hpo_data = case_data['simul-free'].sel(epoch=max_epoch, order=order)  # HP-fixed
    case_stats['means'][SIMUL_LABEL] = hpo_data[objectives[key]].mean(dim='order').values
    case_stats['stds'][SIMUL_LABEL] = hpo_data[objectives[key]].std(dim='order').values * STD_OFFSET
    print(float(hpo_data[objectives[key]].std(dim='order').values.std()))

    if 'weights_init' in case_data:
        print(key, 'WEIGHTS!')
        # import pdb
        # pdb.set_trace()
        order = range(min(NUM_REPLICATES, len(case_data['weights_init'].order)))
        hpo_data = case_data['weights_init'].sel(epoch=max_epoch, order=order)  # HP-fixed
        case_stats['means'][WEIGHTS_LABEL] = hpo_data[objectives[key]].mean(dim='order').values
        case_stats['stds'][WEIGHTS_LABEL] = hpo_data[objectives[key]].std(dim='order').values * STD_OFFSET
        print(float(hpo_data[objectives[key]].std(dim='order').values.std()))
    else:
        case_stats['means'][WEIGHTS_LABEL] = case_stats['means'][SIMUL_LABEL]
        case_stats['stds'][WEIGHTS_LABEL] = case_stats['means'][SIMUL_LABEL]

    if 'bootstrap' in case_data:
        print(key, 'BOOTSTRAP!')
        order = range(min(NUM_REPLICATES, len(case_data['bootstrap'].order)))
        hpo_data = case_data['bootstrap'].sel(epoch=max_epoch, order=order)  # HP-fixed
        case_stats['means'][BOOTSTRAP_LABEL] = hpo_data[objectives[key]].mean(dim='order').values
        case_stats['stds'][BOOTSTRAP_LABEL] = hpo_data[objectives[key]].std(dim='order').values * STD_OFFSET
        print(float(hpo_data[objectives[key]].std(dim='order').values.std()))
    else:
        case_stats['means'][BOOTSTRAP_LABEL] = case_stats['means'][SIMUL_LABEL]
        case_stats['stds'][BOOTSTRAP_LABEL] = case_stats['means'][SIMUL_LABEL]


elapsed_time = time.clock() - start
print(f'It took {elapsed_time}s to compute stats.')


colors = dict(zip(rep_types, ['#1f77ba', '#ff7f0e', '#2ca02c', '#9467bd', '#9467bd', '#9467bd']))

fig = plt.figure(figsize=(7, 2.5))
axes = fig.subplots(2, len(stats),
                    gridspec_kw={'left': .12,
                        'top': .8, 'right': .8, 'bottom': .15,
                        'hspace': 0.7, 'wspace': 0.1}
                    )
axes = axes.T
sns.despine()



def bootstrap_std(data, num_points=20):
    rng = numpy.random.RandomState(1)
    print(data.shape)
    stds = numpy.zeros(num_points)
    for i in range(num_points):
        idx = rng.choice(numpy.arange(data.shape[0]), size=NUM_REPLICATES, replace=True)
        print(data[idx].std())
        print(idx)
        stds[i] = data[idx].std() * STD_OFFSET

    return stds


start = time.time()
for ax, (key, case_data) in zip(axes, stats.items()):
    ax[0].text(0, 1, LABELS[key], 
               horizontalalignment='left',
               verticalalignment='bottom',
               transform=ax[0].transAxes)
    for i, stat_type in enumerate(['means', 'stds']):
        # ax[i].axvline(case_data[stat_type][IDEAL_LABEL][0], 0, 1, color='darkred')

        for j, rep_type in enumerate(rep_types):
            # if rep_type == IDEAL_LABEL and stat_type == 'means':
            #     data = case_data['points']
            if rep_type == IDEAL_LABEL and stat_type == 'stds':
                # data = bootstrap_std(case_data['points'])
                data = numpy.array([])
            else:
                data = case_data[stat_type][rep_type]

            if key not in IGNORE:
                ax[i].scatter(
                    data,
                    j + 1.35 + data * 0,
                    color='k', s=1,
                    marker='d', 
                    alpha=1.0)
                    # marker='|' if rep_type == IDEAL_LABEL else 'd', 
                    # alpha=0.2 if rep_type == IDEAL_LABEL else 1.0)

        if key not in IGNORE:
            boxplots = ax[i].boxplot(
                [case_data[stat_type][rep_type] for rep_type in rep_types],
                labels=rep_types,
                vert=False, widths=.5,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color='k'),
                )
            boxes = boxplots['boxes']

            for box, label in zip(boxes, rep_types):
                box.set_facecolor(colors[label])
        # ax.yaxis.tick_right()
        # ax.set_ylim(.6, 3.48)
        sns.despine(right=True, bottom=False, left=ax[0] is not axes[0, 0], ax=ax[i])
        if ax[0] is not axes[0, 0]:
            ax[i].get_yaxis().set_visible(False)
        ax[i].tick_params(axis='x', which='both', labelsize=X_TICKS_FONT_SIZE)
        ax[i].xaxis.set_major_locator(plt.MaxNLocator(2))
    

# fig.text(.81, 0.85, 'Means')
# fig.text(0.77, .85, "Ideal", fontsize=X_TICKS_FONT_SIZE)
# bbox_props = None # dict(boxstyle="larrow,pad=0.5", fc="w", ec="k", lw=2)
# arrowprops= dict(facecolor='black', shrink=0.05, headwidth=0.5)
# axes[-1, 0].annotate('Ideal',
#             xy=(0.75, 0.80),
#             xytext=(0.78, 0.85),
#             xycoords='figure fraction',
#             horizontalalignment='left',
#             verticalalignment='bottom',
#             bbox=bbox_props,
#             arrowprops=arrowprops)
# fig.text(0.77, .85, "Ideal", fontsize=X_TICKS_FONT_SIZE)
fig.text(0.81, .65, "Average of\ntest performances")
fig.text(0.81, .20, "Standard deviation of\ntest performances")
fig.text(0.81, .01, "* 10e-2", fontsize=X_TICKS_FONT_SIZE)
# fig.text(.81, 0.58, 'Biased')
# fig.text(.81, 0.32, 'Simulation')

# plt.tight_layout(pad=.01)
# plt.subplots_adjust(left=0.5, bottom=0, top=1, right=0.9)
plt.savefig('simul.png', dpi=300)
