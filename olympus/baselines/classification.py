from argparse import ArgumentParser, Namespace

from olympus.datasets import DataLoader, known_datasets, Dataset, SplitDataset
from olympus.metrics import Accuracy
from olympus.observers import ElapsedRealTime

from olympus.models import Model, known_models
from olympus.models.inits import known_initialization

from olympus.optimizers import Optimizer, known_optimizers, LRSchedule, known_schedule

from olympus.tasks import Classification
from olympus.tasks.hpo import HPO, fidelity

from olympus.utils import fetch_device, Chrono, set_verbose_level, select, show_dict
from olympus.utils.functional import flatten
from olympus.utils.options import option
from olympus.utils.storage import StateStorage
from olympus.utils.tracker import TrackLogger

DEFAULT_EXP_NAME = 'classification_{dataset}_{model}_{optimizer}_{lr_scheduler}_{weight_init}'
base = option('base_path', '/tmp/olympus')


def arguments(task=None):
    parser = ArgumentParser(prog='classification', description='Classification Baseline')

    parser.add_argument(
        '--experiment-name', type=str, default=DEFAULT_EXP_NAME,  metavar='EXP_NAME',
        help='Name of the experiment in Orion storage (default: {})'.format(DEFAULT_EXP_NAME))
    parser.add_argument(
        '--model', type=str, metavar='MODEL_NAME', choices=known_models(), required=True,
        help='Name of the model')
    parser.add_argument(
        '--dataset', type=str, metavar='DATASET_NAME',
        choices=known_datasets('classification', include_unknown=True), required=True,
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
        '-v', '--verbose', type=int, default=1,
        help='verbose level'
             '0 disable all progress output, '
             '1 enables progress output, '
             'higher enable higher level logging')
    parser.add_argument(
        '--database', type=str, default=f'file:{base}/classification_baseline.json',
        help='where to store metrics and intermediate results')
    parser.add_argument(
        '--orion-database', type=str, default=None,
        help='where to store Orion data')

    if task is not None:
        hyperparameters = flatten(task.get_space())

        for name, space in hyperparameters.items():
            name = name.replace('.', '-')
            parser.add_argument('--' + name, help=space)

    return parser


chrono = Chrono()


def classification_baseline(model, weight_init,
                            optimizer, lr_scheduler,
                            dataset, batch_size, device,
                            split_method='original',
                            sampler_seed=0, model_seed=0, storage=None, half=False, hpo_done=False,
                            logger=None, validate=True, **config):

    dataset = SplitDataset(
        Dataset(dataset, path=f'{base}/data'),
        split_method=split_method
    )

    loader = DataLoader(
        dataset,
        sampler_seed=sampler_seed,
        batch_size=batch_size
    )

    input_size, target_size = loader.get_shapes()

    model = Model(
        model,
        input_size=input_size,
        output_size=target_size[0],
        weight_init=weight_init,
        seed=model_seed,
        half=half)

    optimizer = Optimizer(optimizer, half=half)

    lr_schedule = LRSchedule(lr_scheduler)

    train, valid = loader.get_train_valid_loaders(hpo_done=hpo_done)

    main_task = Classification(
        classifier=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedule,
        dataloader=train,
        device=device,
        storage=storage,
        logger=logger)

    if validate:
        with chrono.time('metric'):
            main_task.metrics.append(
                Accuracy(name='validation', loader=valid)
            )

    return main_task


def main(**kwargs):
    args = Namespace(**kwargs)
    set_verbose_level(args.verbose)

    device = fetch_device()
    experiment_name = args.experiment_name.format(**kwargs)

    client = TrackLogger(experiment_name, storage_uri=args.database)

    # save partial results here
    state_storage = StateStorage(folder=option('state.storage', f'{base}/classification'))

    def main_task():
        return classification_baseline(device=device, logger=client, storage=state_storage, **kwargs)

    hpo = HPO(
        experiment_name,
        task=main_task,
        algo='ASHA',
        seed=1,
        num_rungs=5,
        num_brackets=1,
        max_trials=300,
        storage=select(args.orion_database, f'track:{args.database}')    # 'legacy:pickleddb:my_data.pkl'
    )
    hpo.metrics.append(ElapsedRealTime())

    hpo.fit(epochs=fidelity(args.epochs), objective='validation_accuracy')

    # Train using train+valid for the final result
    final_task = classification_baseline(device=device, logger=client, storage=state_storage, **kwargs, hpo_done=True)

    if option('worker.id', 0, type=int) == 0:
        hpo.wait_done()

        if hpo.is_broken():
            return

        params = hpo.best_trial.params
        task_args = params.pop('task')

        final_task.init(**params)
        final_task.fit(epochs=args.epochs)

        final_task.finish()

        print('=' * 40)
        print('Final Trial Results')
        show_dict(flatten(params))
        final_task.report(pprint=True, print_fun=print)
        print('=' * 40)

    print('HPO Report')
    print('-' * 40)
    hpo.metrics.report()
    print('=' * 40)


if __name__ == '__main__':
    main(**vars(arguments().parse_args()))
