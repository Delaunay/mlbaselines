import os

from orion.client import create_experiment

from olympus.datasets import DataLoader, merge_data_loaders
from olympus.metrics import Accuracy, ProgressView
from olympus.models import Model, known_models
from olympus.optimizers import Optimizer
from olympus.optimizers.schedules import LRSchedule
from olympus.tasks import Classification
from olympus.utils import fetch_device
from olympus.utils.storage import StateStorage

# import mahler.client as mahler
# from mahler.core.utils.errors import SignalInterruptTask, SignalSuspend


def create_task(dataset, model, optimizer, lr_scheduler, init, epochs, model_seed=1, sampler_seed=1,
                batch_size=128, half=True, merge_train_val=False, folder='.'):

    device = fetch_device()

    loader = DataLoader(
        dataset,
        seed=sampler_seed,
        sampling_method={'name': 'original'},
        batch_size=batch_size
    )

    train_loader = loader.train()
    valid_loader = loader.valid()

    if merge_train_val:
        train_loader = merge_data_loaders(loader.train(), loader.valid())
        valid_loader = loader.test()

    model = Model(
        model,
        half=half,
        input_size=loader.datasets.input_shape,
        output_size=loader.datasets.target_shape[0]
    )

    optimizer = Optimizer(optimizer, model.parameters())
    lr_schedule = LRSchedule(lr_scheduler, optimizer)

    task = Classification(
        classifier=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedule,
        dataloader=train_loader,
        device=device,
        storage=StateStorage(folder=folder))

    task.summary()

    task.metrics.append(Accuracy(name='validation', loader=valid_loader))

    return task


def _get_trial_folder(folder, trial, epochs):
    # Little hack to get id of what would be the trial in last rung
    # (so that we can resume trials across rungs)
    conf = trial.to_dict()
    for param in conf['params']:
        if param['name'] == 'epochs':
            param['value'] = epochs
    trial_id = type(trial)(**conf).id

    return os.path.join(folder, trial_id)


def run_trial(experiment_name, trial_id, **kwargs):
    experiment = create_experiment(experiment_name)
    trial = experiment.get_trial(uid=trial_id)
    experiment.reserve(trial)

    folder = os.environ.get('OLYMPUS_ROOT', '.')
    trial_folder = _get_trial_folder(folder, trial, kwargs['epochs'])

    kwargs.update(trial.params)

    return train(**kwargs)


def train(dataset, model, optimizer, lr_scheduler, init, epochs,
          model_seed=1, sampler_seed=1, merge_train_val=False,
          folder='.', **kwargs):

    task = create_task(dataset, model, optimizer, lr_scheduler, merge_train_val, sampler_seed,
                       folder)

    device = fetch_device()

    task.to(device)

    task.apply(kwargs)

    # This should include model.init and model seeding
    task.init()
    
    # Maybe move the init()?
    task.resume()

    task.fit(epochs)

    # push the latest metrics
    task.finish()

    task.report(pprint=True, print_fun=print)


def create_trials(dataset, model, optimizer, lr_scheduler, init, epochs,
                  model_seed=1, sampler_seed=1, batch_size=128, half=True):

    hpo = HPO(
        create_task(dataset, model, optimizer, lr_scheduler, init, epochs, model_seed, sampler_seed, batch_size, half,
                    merge_train_val=False)
    )

    trial = hpo.step()

    # if trial is None and not (experiment.is_done or experiment.is_broken):
    #     raise SignalInterruptTask('Cannot sample a new trial new')

    hpo.experiment.release(trial)

    # mahler_client = mahler.Client()

    # mahler_client.register(run_trial.delay(**kwargs))

    run_trial(hpo.experiment.name, trial.id,
              dataset=dataset, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
              sampler_seed=sampler_seed, batch_size=batch_size, half=half,
              merge_train_val=False)

    # if not experiment.is_done:
    #     raise SignalInterruptTask('HPO not completed')


if __name__ == '__main__':
    for i in range(10):
        create_trials('mnist', 'logreg', 'sgd', 'none', False, sampler_seed)
