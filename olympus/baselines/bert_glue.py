import argparse
import logging
import os
import shutil

import torch

from olympus.observers.msgtracker import metric_logger

from olympus.baselines.classification import classification_baseline
from olympus.utils.options import options, set_option
from olympus.utils.storage import StateStorage
from olympus.utils import fetch_device, show_dict, set_seeds

logger = logging.getLogger(__name__)


def main(task='rte', bootstrapping_seed=1, sampler_seed=1, init_seed=1, global_seed=1,
         learning_rate=0.00002, beta1=0.9, beta2=0.999, weight_decay=0.0,
         attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1,
         batch_size=32, weight_init='normal',
         warmup=0,
         init_std=0.2, epoch=3, half=False, hpo_done=False,
         uid=None, experiment_name=None, client=None, clean_on_exit=True,
         _interrupt=0):

    print('seeds: init {} / global {} / sampler {} / bootstrapping {}'.format(
        init_seed, global_seed, sampler_seed, bootstrapping_seed))

    base_folder = options('state.storage', '/tmp/storage')
    storage = StateStorage(folder=base_folder, time_buffer=5 * 60)

    task = classification_baseline(
        "bert-{}".format(task), 'normal', 'adam',
        schedule='warmup',
        dataset="glue-{}".format(task),
        sampling_method='original',
        sampler_seed=sampler_seed, init_seed=init_seed,
        batch_size=batch_size, device=fetch_device(),
        storage=storage, half=half, hpo_done=hpo_done, verbose=False, validate=True)

    hyperparameters = dict(
        model={
            'initializer': {'mean': 0.0, 'std': init_std},
            'attention_probs_dropout_prob': attention_probs_dropout_prob,
            'hidden_dropout_prob': hidden_dropout_prob},
        optimizer={
            'lr': learning_rate,
            'beta1': beta1,
            'beta2': beta2,
            'weight_decay': weight_decay},
        lr_schedule={
            'warmup_steps': warmup,
            'max_steps': epoch * len(task.dataloader),
            'iterations': 'step'})

    show_dict(hyperparameters)

    if client is not None:
        task.metrics.append(metric_logger(client=client, experiment=experiment_name))

    if _interrupt:
        from olympus.studies.repro.main import InterruptingMetric
        # Will raise interrupt every `_interrupt` epochs
        task.metrics.append(InterruptingMetric(frequency_epoch=_interrupt))
        storage.time_buffer = 0

    task.init(uid=uid, **hyperparameters)

    # NOTE: Seed global once all special inits are done.
    set_seeds(global_seed)

    task.fit(epochs=epoch)

    # Remove checkpoint
    if clean_on_exit:
        file_path = storage._file(uid)
        try:
            os.remove(file_path)
            print('Removed checkpoint at', file_path)
        except FileNotFoundError:
            print('No checkpoint at ', file_path)
   
    return task.metrics.value().get('validation_error_rate', None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='either rte or sst2', required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--seed-init', type=int, default=1)
    parser.add_argument('--seed-bootstrapping', type=int, default=1)
    parser.add_argument('--seed-sampler', type=int, default=1)
    parser.add_argument('--seed-global', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--attention-dropout', type=float, default=0.1)
    parser.add_argument('--hidden-dropout', type=float, default=0.1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--redirect-log', help='will intercept any stdout/err and log it',
                        action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    set_option('model.cache', 'cache')
    set_option('state.storage', 'storage')

    main(task=args.task, bootstrapping_seed=args.seed_bootstrapping, sampler_seed=args.seed_sampler,
         init_seed=args.seed_init, global_seed=args.seed_global, learning_rate=args.lr,
         attention_probs_dropout_prob=args.attention_dropout,
         hidden_dropout_prob=args.hidden_dropout,
         weight_decay=args.weight_decay, batch_size=args.batch_size, weight_init='normal',
         warmup=args.warmup, init_std=0.2, epoch=args.epoch, half=args.fp16, hpo_done=True)
