import os

from olympus.observers.msgtracker import metric_logger

from olympus.baselines.classification import classification_baseline
from olympus.utils.options import options
from olympus.utils.storage import StateStorage
from olympus.utils import fetch_device, show_dict


def main(bootstrapping_seed=1, sampler_seed=1, transform_seed=1, init_seed=1,
         learning_rate=0.1, momentum=0.9, weight_decay=5e-4, gamma=0.99,
         weight_init='glorot_uniform',
         epoch=120, half=False, hpo_done=False,
         uid=None, experiment_name=None, client=None, clean_on_exit=True,
         _interrupt=0):

    base_folder = options('state.storage', '/tmp')
    storage = StateStorage(folder=base_folder, time_buffer=5 * 60)
    print(base_folder)

    sampling_method = {
        'split_method': 'bootstrap',
        'ratio': 0.1666,
        'seed': bootstrapping_seed,
        'balanced': True}

    batch_size=128

    task = classification_baseline(
        'vgg11', 'glorot_uniform', 'sgd', schedule='exponential',
        dataset='cifar10', batch_size=batch_size, device=fetch_device(),
        data_augment=True, split_method=sampling_method,
        sampler_seed=sampler_seed, transform_seed=transform_seed, init_seed=init_seed,
        storage=storage, half=half, hpo_done=hpo_done, verbose=False,
        validate=True)

    hyperparameters = dict(
        model={'initializer': {'gain': 1.0}},
        optimizer=dict(
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay),
        lr_schedule=dict(
            gamma=gamma))

    show_dict(hyperparameters)

    if client is not None:
        task.metrics.append(metric_logger(client=client, experiment=experiment_name))

    if _interrupt:
        from studies import InterruptingMetric
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

    return float(task.metrics.value()['validation_error_rate'])


if __name__ == '__main__':
    main(epoch=2)
