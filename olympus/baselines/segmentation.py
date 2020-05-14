import os
import torch
import torch.nn as nn

import numpy as np

from olympus.datasets import DataLoader, known_datasets, Dataset, SplitDataset
from olympus.metrics import MeanIoU

from olympus.models import Model, known_models, Initializer, known_initialization
from olympus.optimizers import Optimizer, known_optimizers, LRSchedule

from olympus.tasks import Segmentation
from olympus.utils import (
        fetch_device, set_seeds, get_parameters, show_dict)

from olympus.observers import metric_logger
from olympus.utils.options import option, options
from olympus.utils.storage import StateStorage


DEFAULT_EXP_NAME = 'segmentation_{dataset}_{model}_{optimizer}_{initializer}'


def segmentation_baseline(model, initializer, optimizer,
                          dataset, batch_size, device,
                          split_method='original',
                          sampler_seed=0,
                          init_seed=0,
                          global_seed=0, storage=None, half=False, hpo_done=False,
                          data_path='/tmp/olympus',
                          validate=True, hyper_parameters=None, uri_metric=None,
                          valid_batch_size=None,
                          **config):
    set_seeds(global_seed)

    # dataset size: 2913
    dataset = SplitDataset(
        Dataset(dataset, path=option('data.path', data_path)),
        split_method=split_method,
    )

    loader = DataLoader(
        dataset,
        sampler_seed=sampler_seed,
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        pin_memory=True,
        num_workers=0,
    )

    input_size, target_size = loader.get_shapes()

    init = Initializer(
        initializer,
        seed=init_seed,
        **get_parameters('initializer', hyper_parameters)
    )

    model = Model(
        model,
        input_size=input_size,
        output_size=target_size[0],
        weight_init=init,
        half=half)

    optimizer = Optimizer(
        optimizer, half=half, **get_parameters('optimizer', hyper_parameters)
    )

    lr_schedule = LRSchedule('none', **get_parameters('schedule', hyper_parameters))

    train, valid, test = loader.get_loaders(hpo_done=hpo_done)

    additional_metrics = []

    if validate and valid:
        additional_metrics.append(
            MeanIoU(name='validation', loader=valid)
        )

    if validate and test:
        additional_metrics.append(
            MeanIoU(name='test', loader=test)
        )

    def get_label_counts(dataloader):
        cumulative_counts = {}
        print('get_label_counts(): ', end='')
        for i, (_, labels) in enumerate(dataloader, 1):
            if labels.device.type == 'cuda':
                labels = labels.cpu()
            unique, counts = np.unique(labels.numpy(), return_counts=True)
            for u, c in zip(unique, counts):
                if u not in cumulative_counts:
                    cumulative_counts[u] = 0
                cumulative_counts[u] += c
            if i % (len(dataloader) // 10) == 0:
                print('{}%... '.format(100 * i // len(dataloader)), end='')
        print()
        return cumulative_counts

    def get_criterion_weight(counts, ignore_index=255):
        counts = counts.copy()
        if ignore_index in counts:
            del counts[ignore_index]
        total_count = sum([counts[unique] for unique in sorted(counts)])
        weight  = np.array([total_count / counts[unique] for unique in sorted(counts)], dtype=np.float32)
        weight /= weight.size
        return weight

    nclasses = 21
    counts = get_label_counts(train)
    weight = get_criterion_weight(counts)
    weight = torch.tensor(weight)
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=255)

    main_task = Segmentation(
        model, optimizer, lr_schedule, train, criterion, nclasses,
        device=device, storage=storage, metrics=additional_metrics)

    return main_task


def main(bootstrapping_seed=1, sampler_seed=1, init_seed=1,
         batch_size=16, learning_rate=0.001, momentum=0.9,
         weight_decay=1e-4, epoch=240, half=False, hpo_done=False,
         uid=None, experiment_name=None, client=None, clean_on_exit=True,
         _interrupt=0):

    base_folder = options('state.storage', '/tmp')
    storage = StateStorage(folder=base_folder, time_buffer=5 * 60)

    split_method = {
        'split_method': 'bootstrap',
        'ratio': 0.25,  # This means 50% training, 25% valid, 25% test
        'seed': bootstrapping_seed,
        'balanced': False}

    task = segmentation_baseline('fcn_resnet18', 'self_init', 'SGD',
            dataset='voc-segmentation', batch_size=batch_size, device=fetch_device(),
            split_method=split_method, sampler_seed=sampler_seed, init_seed=init_seed,
            storage=storage, half=half, hpo_done=hpo_done, verbose=False, validate=True)

    hyperparameters = {
            'model': {
                'initializer': {
                    'gain': 1.0}},
            'optimizer': {
                'lr': learning_rate,
                'momentum': momentum,
                'weight_decay': weight_decay}}
    show_dict(hyperparameters)

    if client is not None:
        task.metrics.append(metric_logger(client=client, experiment=experiment_name))

    if _interrupt:
        from olympus.studies.repro.main import InterruptingMetric
        # Will raise interrupt every `_interrupt` epochs
        task.metrics.append(InterruptingMetric(frequency_epoch=_interrupt))
        storage.time_buffer = 0

    task.init(uid=uid, **hyperparameters)
    task.fit(epochs=epoch)

    # Remove checkpoint
    if clean_on_exit:
        file_path = storage._file(uid)
        try:
            os.remove(file_path)
            print('Removed checkpoint at', file_path)
        except FileNotFoundError:
            print('No checkpoint at ', file_path)

    show_dict(task.metrics.value())

    return float(task.metrics.value()['validation_mean_jaccard_distance'])


if __name__ == '__main__':
    main(epoch=6)
