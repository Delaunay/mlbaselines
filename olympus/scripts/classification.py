import argparse
import os

import torch
import torch.utils.data

from orion.client import create_experiment

from olympus.hpo import TrialIterator
from olympus.datasets import build_loaders, merge_data_loaders
from olympus.datasets import factories as dataset_factories
import olympus.distributed.multigpu as distributed
from olympus.metrics import Accuracy
from olympus.models import build_model
from olympus.models import factories as model_factories
from olympus.tasks import Classification
from olympus.optimizers import get_optimizer_builder
from olympus.optimizers import factories as optimizer_factories
from olympus.utils import task_arguments, get_storage, show_dict, fetch_device
from olympus.utils.storage import StateStorage


DEFAULT_EXP_NAME = 'classification_{dataset}_{model}_{optimizer}'


def arguments(subparsers=None):
    parser = task_arguments('classification', 'Classification script', subparsers)

    parser.add_argument(
        '--experiment-name', type=str, default=DEFAULT_EXP_NAME,  metavar='EXP_NAME',
        help='Name of the experiment in Orion storage (default: {})'.format(DEFAULT_EXP_NAME))
    parser.add_argument(
        '--model', type=str, metavar='MODEL_NAME', choices=model_factories.keys(), required=True,
        help='Name of the model')
    parser.add_argument(
        '--dataset', type=str, metavar='DATASET_NAME', choices=dataset_factories.keys(), required=True,
        help='Name of the dataset')
    parser.add_argument(
        '--optimizer', type=str, default='sgd',
        metavar='OPTIMIZER_NAME', choices=optimizer_factories.keys(),
        help='Name of the optimiser (default: sgd)')
    parser.add_argument(
        '--batch-size', type=int, default=128, metavar='N',
        help='input batch size for training (default: 128)')
    parser.add_argument(
        '--epochs', type=int, default=300, metavar='N',
        help='maximum number of epochs to train (default: 300)')
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
    parser.add_argument(
        '--half', action='store_true', default=False,
        help='enable fp16 training'
    )

    # add the arguments related to distributed training
    return distributed.arguments(parser)


def train(dataset, model, optimizer, epochs, merge_train_val=False, folder='.', **kwargs):

    # Apply Orion overrides
    distributed.enable_distributed_process(
        kwargs.get('rank'),
        kwargs.get('dist_url'),
        kwargs.get('world_size')
    )

    device = fetch_device()

    datasets, loaders = build_loaders(
        dataset,
        sampling_method={'name': 'original'},
        batch_size=kwargs['batch_size']
    )

    train_loader = loaders['train']
    valid_loader = loaders['valid']

    if merge_train_val:
        train_loader = merge_data_loaders(train_loader, valid_loader)
        valid_loader = loaders['test']

    model = build_model(
        model,
        half=kwargs.get('half', False),
        input_size=datasets.input_shape,
        output_size=datasets.output_shape[0]
    ).to(device)

    # NOTE: Some model have specific way of building for distributed computing
    #       (i.e. large output layers) This may be better integrated in the model builder.
    model = distributed.data_parallel(model)

    optimizer_builder = get_optimizer_builder(optimizer)

    task = Classification(
        classifier=model,
        optimizer=optimizer_builder(
            model.parameters(),
            weight_decay=kwargs['weight_decay'],
            half=kwargs.get('half', False),
            **optimizer_builder.get_params(kwargs)),
        device=device,
        storage=StateStorage(folder=folder))

    # task.metrics.append(Accuracy(name='training', loader=train_loader))
    task.metrics.append(Accuracy(name='validation', loader=valid_loader))

    task.resume()

    # TODO: What is supposed to be the context?
    task.fit(train_loader, epochs, {})

    # push the latest metrics
    task.finish()

    task.report(pprint=True, print_fun=print)

    return task.metrics.value()['validation_accuracy']


def get_trial_folder(folder, trial, epochs):
    # Little hack to get id of what would be the trial in last rung
    # (so that we can resume trials across rungs)
    conf = trial.to_dict()
    for param in conf['params']:
        if param['name'] == 'epochs':
            param['value'] = epochs
    trial_id = type(trial)(**conf).id

    return os.path.join(folder, trial_id)


def main(experiment_name, dataset, model, optimizer, epochs, folder='.', **kwargs):

    for key in ['dataset', 'model', 'optimizer']:
        experiment_name = experiment_name.replace('{' + key + '}', locals()[key])

    space = {
        'weight_decay': 'loguniform(1e-10, 1e-3)',
        'epochs': f'fidelity(1, {epochs}, base=4)'
    }

    optimizer_builder = get_optimizer_builder(optimizer)

    space.update(optimizer_builder.get_space())

    experiment = create_experiment(
        name=experiment_name,
        max_trials=50,
        space=space,
        algorithms={
            'asha': {
                'seed': 1,
                'num_rungs': 5,
                'num_brackets': 1
            }},
        storage=get_storage('legacy:pickleddb:test.pkl')
    )

    experiment_folder = os.path.join(folder, 'classification', experiment_name)

    for trial in TrialIterator(experiment):
        trial_folder = get_trial_folder(experiment_folder, trial, epochs)

        show_dict(trial.params)
        kwargs.update(trial.params)

        validation_accuracy = train(dataset, model, optimizer, folder=trial_folder, **kwargs)

        experiment.observe(
            trial,
            [dict(name='ValidationErrorRate', value=1 - validation_accuracy, type='objective')])

    if experiment.is_broken:
        raise RuntimeError('Experiment is broken!')

    trial = experiment.get_trial(uid=experiment.stats['best_trials_id'])

    kwargs.update(trial.params)

    print('Training with best hyper-parameters on train+valid.')
    show_dict(trial.params)
    test_accuracy = train(dataset, model, optimizer, **kwargs)

    # TODO: Find a way so register this
    print('Test accuracy: ', test_accuracy)


if __name__ == '__main__':
    main(**arguments().parse_args())
