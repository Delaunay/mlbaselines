from argparse import ArgumentParser, Namespace

from olympus.datasets import DataLoader, known_datasets
from olympus.models import Model, known_models
from olympus.models.inits import known_initialization
from olympus.optimizers import Optimizer, known_optimizers
from olympus.optimizers.schedules import LRSchedule, known_schedule
from olympus.tasks import ObjectDetection
from olympus.tasks.hpo import HPO, fidelity
from olympus.utils import fetch_device, Chrono, set_verbose_level
from olympus.utils.options import option
from olympus.utils.storage import StateStorage
from olympus.utils.tracker import TrackLogger
from olympus.metrics import Loss


DEFAULT_EXP_NAME = 'detection_{dataset}_{model}_{optimizer}_{lr_scheduler}_{weight_init}'


def arguments():
    parser = ArgumentParser(prog='detection', description='Detection Baseline')

    parser.add_argument(
        '--experiment-name', type=str, default=DEFAULT_EXP_NAME,  metavar='EXP_NAME',
        help='Name of the experiment in Orion storage (default: {})'.format(DEFAULT_EXP_NAME))
    parser.add_argument(
        '--model', type=str, metavar='MODEL_NAME', choices=known_models(), required=True,
        help='Name of the model')
    parser.add_argument(
        '--dataset', type=str, metavar='DATASET_NAME', choices=known_datasets('detection'), required=True,
        help='Name of the dataset')
    parser.add_argument(
        '--optimizer', type=str, default='sgd',
        metavar='OPTIMIZER_NAME', choices=known_optimizers(),
        help='Name of the optimiser (default: sgd)')
    parser.add_argument(
        '--lr-scheduler', type=str, default='none',
        metavar='LR_SCHEDULER_NAME', choices=known_schedule(),
        help='Name of the lr scheduler (default: none)')
    parser.add_argument(
        '--weight-init', type=str, default='glorot_uniform',
        metavar='INIT_NAME', choices=known_initialization(),
        help='Name of the initialization (default: glorot_uniform)')
    parser.add_argument(
        '--batch-size', type=int, default=128, metavar='N',
        help='input batch size for training (default: 128)')
    parser.add_argument(
        '--epochs', type=int, default=300, metavar='N',
        help='maximum number of epochs to train (default: 300)')
    parser.add_argument(
        '--model-seed', type=int, default=1,
        help='random seed for model initialization (default: 1)')
    parser.add_argument(
        '--sampler-seed', type=int, default=1,
        help='random seed for sampler during iterations (default: 1)')
    parser.add_argument(
        '--half', action='store_true', default=False,
        help='enable fp16 training')
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='verbose level')

    return parser


chrono = Chrono()


def reduce_loss(loss_dict):
    return sum(loss for loss in loss_dict.values())


def detection_baseline(model, weight_init,
                       optimizer, lr_scheduler,
                       dataset, batch_size, device,
                       sampler_seed=0, model_seed=0, storage=None, half=False, hpo_done=False, logger=None, **config):

    with chrono.time('loader'):
        dataset = DataLoader(
            dataset,
            seed=sampler_seed,
            sampling_method={'name': 'original'},
            batch_size=batch_size)

    with chrono.time('get_shapes'):
        input_size, target_size = dataset.get_shapes()

    with chrono.time('model'):
        model = Model(
            model,
            input_size=input_size,
            output_size=dataset.datasets.num_classes,
            weight_init=weight_init,
            seed=model_seed,
            half=half)

    with chrono.time('optimizer'):
        optimizer = Optimizer(optimizer, half=half)

    with chrono.time('lr'):
        lr_schedule = LRSchedule(lr_scheduler)

    with chrono.time('get_loaders'):
        train, valid = dataset.get_train_valid_loaders(hpo_done)

    with chrono.time('task'):
        main_task = ObjectDetection(
            detector=model,
            optimizer=optimizer,
            lr_scheduler=lr_schedule,
            dataloader=train,
            device=device,
            storage=storage,
            criterion=reduce_loss,
            logger=logger)

    with chrono.time('metric'):
        main_task.metrics.append(
            Loss(name='validation', loader=valid)
        )

    return main_task


def main(**kwargs):
    args = Namespace(**kwargs)
    set_verbose_level(args.verbose)

    device = fetch_device()
    experiment_name = args.experiment_name.format(**kwargs)

    path = option('trial.storage', 'file://track_test.json')
    client = TrackLogger(experiment_name, storage_uri=path)

    # save partial results here
    state_storage = StateStorage(
        folder=option('state.storage', '/tmp'),
        time_buffer=30)

    def main_task():
        return detection_baseline(device=device, logger=client, storage=state_storage, **kwargs)

    hpo = HPO(
        experiment_name,
        task=main_task,
        algo='ASHA',
        seed=1,
        num_rungs=5,
        num_brackets=1,
        max_trials=300,
        storage=f'track:{path}'
    )

    hpo.fit(epochs=fidelity(args.epochs), objective='validation_loss')

    # Train using train+valid for the final result
    final_task = detection_baseline(device=device, logger=client, storage=state_storage, **kwargs, hpo_done=True)

    params = hpo.best_trial.params
    task_args = params.pop('task')

    final_task.init(**params)
    final_task.fit(**task_args)

    final_task.finish()

    print('=' * 40)
    print('Results')
    print('-' * 40)
    final_task.report(pprint=True, print_fun=print)
    print('=' * 40)


if __name__ == '__main__':
    main(**vars(arguments().parse_args()))
