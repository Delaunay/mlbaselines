from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms

from orion.client import create_experiment

from olympus.datasets import build_loaders
from olympus.datasets import factories as dataset_factories
import olympus.distributed.multigpu as distributed
from olympus.hpo import OrionClient
from olympus.metrics import ValidationAccuracy
from olympus.models import build_model
from olympus.models import factories as model_factories
from olympus.tasks import Classification
from olympus.optimizers import get_optimizer_builder
from olympus.optimizers import factories as optimizer_factories



DEFAULT_EXP_NAME = 'classification_{dataset}_{model}_{optimizer}'


def arg_parser(subparsers=None):
    description = 'Classification script'
    if subparsers:
        parser = subparsers.add_parser('classification', help=description)
    else:
        parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '--experiment-name', type=str, default=DEFAULT_EXP_NAME,
        metavar='EXP_NAME',
        help='Name of the experiment in Orion storage (default: {})'.format(DEFAULT_EXP_NAME))
    parser.add_argument(
        '--model', type=str,
        metavar='MODEL_NAME', choices=model_factories.keys(),
        help='Name of the model')
    parser.add_argument(
        '--dataset', type=str,
        metavar='DATASET_NAME', choices=dataset_factories.keys(),
        help='Name of the dataset')
    parser.add_argument(
        '--optimizer', type=str,
        metavar='OPTIMIZER_NAME', choices=optimizer_factories.keys(),
        default='sgd',
        help='Name of the optimiser (default: sgd)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='maximum number of epochs to train (default: 300)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    return parser


def parse_args(argv=None):
    return arg_parser().parse_args(argv)


def main(experiment_name, dataset, model, optimizer, epochs, **kwargs):

    experiment_name = experiment_name

    for key in ['dataset', 'model', 'optimizer']:
        experiment_name = experiment_name.replace('{' + key + '}', locals()[key])

    space = {
            'weight_decay': 'loguniform(1e-10, 1e-3)',
            'epochs': 'fidelity(1, {}, base=4)'.format(epochs)}

    optimizer_builder = get_optimizer_builder(optimizer)

    space.update(optimizer_builder.get_space())

    hpo_client = OrionClient()

    hpo_client.new_trial(
        name=experiment_name,
        space=space,
        algorithms={
            'asha': {
                'seed': 1,
                'num_rungs': 5,
                'num_brackets': 1
            }})

    # Apply Orion overrides
    kwargs = hpo_client.sample(kwargs)
    if 'rank' in kwargs:
        distributed.enable_distributed_process(kwargs['rank'], kwargs['dist_url'], kwargs['world_size'])

    datasets, loaders = build_loaders(dataset, sampling_method={'name': 'original'}, batch_size=kwargs['batch_size'])
    train_loader = loaders['train']
    valid_loader = loaders['valid']

    model = build_model(model, input_size=datasets.input_shape, output_size=datasets.output_shape[0])
    # NOTE: Some model have specific way of building for distributed computing
    #       (i.e. large output layers) This may be better integrated in the model builder.
    if 'rank' in kwargs:
        model = distributed.data_parallel(model)

    task = Classification(
        classifier=model,
        optimizer=optimizer_builder(
            model.parameters(),
            weight_decay=kwargs['weight_decay'],
            **optimizer_builder.get_params(kwargs)))

    task.device = torch.device('cpu')

    task.metrics.append(
        ValidationAccuracy(loader=valid_loader)
    )

    # TODO: What is supposed to be the context?
    task.fit(train_loader, epochs, {})

    # push the latest metrics
    task.finish()

    task.report(pprint=True, print_fun=print)

    hpo_client.report_objective(
        name='ValidationAccuracy',
        value=task.metrics.value()['validation_accuracy'])


if __name__ == '__main__':
    main(**parse_args())