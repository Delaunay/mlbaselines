from argparse import ArgumentParser, Namespace

from olympus.datasets import DataLoader, known_datasets
from olympus.metrics import Accuracy
from olympus.models import Model, known_models
from olympus.models.inits import known_initialization
from olympus.optimizers import Optimizer, known_optimizers
from olympus.optimizers.schedules import LRSchedule, known_schedule
from olympus.tasks import Classification
from olympus.utils import fetch_device, Chrono, set_verbose_level
from olympus.utils.options import options
from olympus.utils.storage import StateStorage
from olympus.utils.tracker import TrackLogger

DEFAULT_EXP_NAME = 'classification_{dataset}_{model}_{optimizer}_{lr_scheduler}_{weight_init}'


def arguments():
    parser = ArgumentParser(prog='classification', description='Classification Baseline')

    parser.add_argument(
        '--experiment-name', type=str, default=DEFAULT_EXP_NAME,  metavar='EXP_NAME',
        help='Name of the experiment in Orion storage (default: {})'.format(DEFAULT_EXP_NAME))
    parser.add_argument(
        '--model', type=str, metavar='MODEL_NAME', choices=known_models(), required=True,
        help='Name of the model')
    parser.add_argument(
        '--dataset', type=str, metavar='DATASET_NAME', choices=known_datasets('classification', include_unknown=True), required=True,
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
        '--database', type=str, default='file://track_test.json',
        help='where to store metrics and intermediate results')

    return parser


def main():
    args, kwargs = arguments().parse_known_args()
    set_verbose_level(args.verbose)

    device = fetch_device()
    experiment_name = args.experiment_name.format(**(vars(args)))

    client = TrackLogger(experiment_name, storage_uri=args.database)
    state_storage = StateStorage(folder=options('state.storage', '/tmp/olympus'), time_buffer=30)
    chrono = Chrono()

    with chrono.time('loader'):
        dataset = DataLoader(
            args.dataset,
            seed=args.sampler_seed,
            sampling_method=args.sampling_method,
            batch_size=args.batch_size)

    with chrono.time('get_shapes'):
        input_size, target_size = dataset.get_shapes()

    with chrono.time('model'):
        model = Model(
            args.model,
            input_size=input_size,
            output_size=target_size[0],
            weight_init=args.weight_init,
            seed=args.model_seed,
            half=args.half)

    with chrono.time('optimizer'):
        optimizer = Optimizer(args.optimizer, half=args.half)

    with chrono.time('lr'):
        lr_schedule = LRSchedule(args.lr_scheduler)

    with chrono.time('get_loaders'):
        train, valid = dataset.get_train_valid_loaders()

    with chrono.time('task'):
        classify = Classification(
            classifier=model,
            optimizer=optimizer,
            lr_scheduler=lr_schedule,
            dataloader=train,
            device=device,
            storage=state_storage,
            logger=client)

    with chrono.time('metric'):
        classify.metrics.append(
            Accuracy(name='validation', loader=valid)
        )

    classify.init(**kwargs)
    classify.fit(epoch=args.epochs)
    classify.finish()

    print('=' * 40)
    print('Results')
    print('-' * 40)
    classify.report(pprint=True, print_fun=print)
    print('=' * 40)


if __name__ == '__main__':
    main()
