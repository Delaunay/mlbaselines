from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter

from olympus.datasets import DataLoader, known_datasets, Dataset, SplitDataset
from olympus.metrics import Accuracy
from olympus.observers import ElapsedRealTime, metric_logger

from olympus.models import Model, known_models, Initializer, known_initialization
from olympus.optimizers import Optimizer, known_optimizers, LRSchedule, known_schedule

from olympus.tasks import Classification
from olympus.hpo import HPOptimizer, Fidelity
from olympus.tasks.hpo import HPO

from olympus.utils import (
    fetch_device, set_verbose_level, show_dict, show_hyperparameter_space, get_parameters,
    set_seeds, required)
from olympus.utils.functional import flatten
from olympus.utils.options import option
from olympus.utils.storage import StateStorage

DEFAULT_EXP_NAME = 'classification_{dataset}_{model}_{optimizer}_{schedule}_{initializer}'


def arguments():
    parser = ArgumentParser(
        prog='classification',
        description='Classification Baseline',
        epilog=show_hyperparameter_space(),
        formatter_class=RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--experiment-name', type=str, default=DEFAULT_EXP_NAME,  metavar='EXP_NAME',
        help='Name of the experiment in Orion storage (default: {})'.format(DEFAULT_EXP_NAME))
    parser.add_argument(
        '--arg-file', type=str, default=None, metavar='ARGS',
        help='Json File containing the arguments to use')
    parser.add_argument(
        '--model', type=str, metavar='MODEL_NAME', choices=known_models(), default=required,
        help='Name of the model')
    parser.add_argument(
        '--dataset', type=str, metavar='DATASET_NAME', default=required,
        choices=known_datasets('classification', include_unknown=True),
        help='Name of the dataset')
    parser.add_argument(
        '--optimizer', type=str, default='sgd',
        metavar='OPTIMIZER_NAME', choices=known_optimizers(),
        help='Name of the optimiser (default: sgd)')
    parser.add_argument(
        '--schedule', type=str, default='none',
        metavar='LR_SCHEDULER_NAME', choices=known_schedule(),
        help='Name of the lr scheduler (default: none)')
    parser.add_argument(
        '--initializer', type=str, default='glorot_uniform',
        metavar='INIT_NAME', choices=known_initialization(),
        help='Name of the weight initialization method (default: glorot_uniform)')
    parser.add_argument(
        '--batch-size', type=int, default=128, metavar='B',
        help='input batch size for training (default: 128)')
    parser.add_argument(
        '--epochs', type=int, default=300, metavar='N',
        help='maximum number of epochs to train (default: 300)')
    parser.add_argument(
        '--min-epochs', type=int, default=20, metavar='MN',
        help='minimum number of epochs to train (default: 20) for HPO')
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
        '--uri', type=str, default=None,
        help='Resource to use to store metrics')
    parser.add_argument(
        '--database', type=str, default='olympus',
        help='which database to use')
    parser.add_argument(
        '--rank', type=int, default=1,
        help='Rank of worker for distributed training')
    parser.add_argument(
        '--world-size', type=int, default=1,
        help='Number of workers for distributed training')
    parser.add_argument(
        '--dist-url', type=str,
        help='Connection URL for distributed training')

    return parser


def classification_baseline(model, initializer,
                            optimizer, schedule,
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

    dataset = SplitDataset(
        Dataset(dataset, path=option('data.path', data_path)),
        split_method=split_method
    )

    loader = DataLoader(
        dataset,
        sampler_seed=sampler_seed,
        batch_size=batch_size,
        valid_batch_size=valid_batch_size
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

    lr_schedule = LRSchedule(schedule, **get_parameters('schedule', hyper_parameters))

    train, valid, test = loader.get_loaders(hpo_done=hpo_done)

    additional_metrics = []
    if validate:
        additional_metrics.append(
            Accuracy(name='validation', loader=valid)
        )

    main_task = Classification(
        classifier=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedule,
        dataloader=train,
        device=device,
        storage=storage,
        metrics=additional_metrics)

    if validate and valid:
        main_task.metrics.append(
            Accuracy(name='validation', loader=valid)
        )

    if validate and test:
        main_task.metrics.append(
            Accuracy(name='test', loader=test)
        )

    return main_task


def main(**kwargs):
    show_dict(kwargs)

    args = Namespace(**kwargs)
    set_verbose_level(args.verbose)

    device = fetch_device()
    experiment_name = args.experiment_name.format(**kwargs)

    # save partial results here
    state_storage = StateStorage(
        folder=option('state.storage', '/tmp/olympus/classification'),
        time_buffer=30)

    def main_task():
        task = classification_baseline(device=device, storage=state_storage, **kwargs)

        if args.uri is not None:
            logger = metric_logger(args.uri, args.database, experiment_name)
            task.metrics.append(logger)

        return task

    space = main_task().get_space()

    # If space is not empty we search the best hyper parameters
    params = {}
    if space:
        show_dict(space)
        hpo = HPOptimizer('hyperband', space=space,
                          fidelity=Fidelity(args.min_epochs, args.epochs).to_dict())

        hpo_task = HPO(hpo, main_task)
        hpo_task.metrics.append(ElapsedRealTime())

        trial = hpo_task.fit(objective='validation_accuracy')
        print(f'HPO is done, objective: {trial.objective}')
        params = trial.params
    else:
        print('No hyper parameter missing, running the experiment...')
    # ------

    # Run the experiment with the best hyper parameters
    # -------------------------------------------------
    if params is not None:
        # Train using train + valid for the final result
        final_task = classification_baseline(device=device, **kwargs, hpo_done=True)
        final_task.init(**params)
        final_task.fit(epochs=args.epochs)

        print('=' * 40)
        print('Final Trial Results')
        show_dict(flatten(params))
        final_task.report(pprint=True, print_fun=print)
        print('=' * 40)


if __name__ == '__main__':
    from olympus.utils import parse_args
    main(**vars(parse_args(arguments())))
