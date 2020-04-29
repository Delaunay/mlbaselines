import os
import shutil

from olympus.observers.msgtracker import metric_logger

from olympus.baselines.classification import classification_baseline
from olympus.utils.options import options, set_option
from olympus.utils.storage import StateStorage
from olympus.utils import fetch_device, show_dict


def main(task='rte', bootstraping_seed=1, sampler_seed=1, init_seed=1,
         learning_rate=0.00002, beta1=0.9, beta2=0.999, weight_decay=0.0,
         batch_size=32, weight_init='normal',
         warmup=0,
         init_std=0.2, epoch=3, half=False, hpo_done=False,
         uid=None, experiment_name=None, client=None, clean_on_exit=True):

    base_folder = options('state.storage', '/tmp/storage')
    storage = StateStorage(folder=base_folder, time_buffer=5 * 60)

    sampling_method = {
        'name': 'bootstrap', 'ratio': 1,
        'split_ratio': 0.1666, 
        'seed': bootstraping_seed}

    task = classification_baseline(
        "bert-{}".format(task), 'normal', 'adam',
        schedule='warmup',
        dataset="glue-{}".format(task),
        # sampling_method=sampling_method,
        sampling_method='original',
        sampler_seed=sampler_seed, init_seed=init_seed,
        batch_size=batch_size, device=fetch_device(),
        storage=storage, half=half, hpo_done=hpo_done, verbose=False, validate=True)

    hyperparameters = dict(
        model={'initializer': {'mean': 0.0, 'std': init_std}},
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
        task.metrics.new_trial(hyperparameters, uid)

    task.init(**hyperparameters)

    task.fit(epochs=epoch)

    # Remove checkpoint
    if clean_on_exit and os.path.exists(base_folder):
        shutil.rmtree(base_folder, ignore_errors=True)
        print('Removed checkpoints at', base_folder)
   
    return float(task.metrics.value()['validation_accuracy'])
    # return float(task.metrics.value()['validation_error_rate'])


if __name__ == '__main__':
    set_option('model.cache', 'cache')
    set_option('state.storage', 'storage')
    main(task='sst2', epoch=3, bootstraping_seed=1, sampler_seed=1, init_seed=4, hpo_done=True)
