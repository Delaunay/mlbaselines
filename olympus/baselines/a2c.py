from argparse import ArgumentParser, Namespace

from olympus.models import Model, known_models
from olympus.models.inits import known_initialization, Initializer

from olympus.reinforcement import Environment
from olympus.reinforcement.dataloader import RLDataLoader, simple_replay_vector
from olympus.observers import ElapsedRealTime
from olympus.optimizers import Optimizer, known_optimizers
from olympus.optimizers.schedules import LRSchedule, known_schedule

from olympus.tasks.reinforcement.a2c import A2C
from olympus.tasks.hpo import HPO, fidelity

from olympus.utils import fetch_device, set_verbose_level, select, show_dict
from olympus.utils.functional import flatten
from olympus.utils.options import option
from olympus.utils.storage import StateStorage
from olympus.utils.tracker import TrackLogger

DEFAULT_EXP_NAME = 'reinforcement_{env_name}_{model}_{optimizer}_{lr_scheduler}_{weight_init}'
base = option('base_path', '/tmp/olympus')


def arguments():
    parser = ArgumentParser(prog='classification', description='Classification Baseline')

    parser.add_argument(
        '--experiment-name', type=str, default=DEFAULT_EXP_NAME, metavar='EXP_NAME',
        help='Name of the experiment in Orion storage (default: {})'.format(DEFAULT_EXP_NAME))
    parser.add_argument(
        '--epochs', type=int, default=300, metavar='N',
        help='maximum number of epochs to train (default: 300)')

    parser.add_argument(
        '--model', type=str, metavar='MODEL_NAME', choices=known_models(), required=True,
        help='Name of the model')
    parser.add_argument(
        '--weight-init', type=str, default='glorot_uniform',
        metavar='INIT_NAME', choices=known_initialization(),
        help='Name of the initialization (default: glorot_uniform)')
    parser.add_argument(
        '--model-seed', type=int, default=1,
        help='random seed for model initialization (default: 1)')

    parser.add_argument(
        '--env-name', type=str, default='SpaceInvaders-v0',
        help='Name of the gym environment')
    parser.add_argument(
        '--parallel-sim', type=int, default=4,
        help='Number of simulation in parallel')
    parser.add_argument(
        '--max-steps', type=int, default=32,
        help='Maximum number of simulation steps')
    parser.add_argument(
        '--num-steps', type=int, default=32,
        help='number of steps for before doing a gradient update')
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='Discount factor')

    parser.add_argument(
        '--optimizer', type=str, default='sgd',
        metavar='OPTIMIZER_NAME', choices=known_optimizers(),
        help='Name of the optimiser (default: sgd)')
    parser.add_argument(
        '--lr-scheduler', type=str, default='none',
        metavar='LR_SCHEDULER_NAME', choices=known_schedule(),
        help='Name of the lr scheduler (default: none)')

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
        '--database', type=str, default=f'file:{base}/a2c_baseline.json',
        help='where to store metrics and intermediate results')
    parser.add_argument(
        '--orion-database', type=str, default=None,
        help='where to store Orion data')

    return parser


def a2c_baseline(env_name, parallel_sim, weight_init, model, model_seed, optimizer,
                 lr_scheduler, num_steps, half, device, logger, storage, **config):
    def to_nchw(states):
        return states.permute(0, 3, 1, 2)

    env = Environment(
        env_name,
        parallel_env=parallel_sim,
        transforms=to_nchw
    )

    init = Initializer(
        weight_init,
        seed=model_seed,
        gain=1.0
    )

    model = Model(
        model,
        input_size=env.input_size,
        output_size=env.target_size[0],
        weight_init=init,
        half=half)

    loader = RLDataLoader(
        env,
        replay=simple_replay_vector(num_steps=num_steps),
        actor=model.act,
        critic=model.critic
    )

    optimizer = Optimizer(optimizer, half=half)

    lr_schedule = LRSchedule(lr_scheduler)

    task = A2C(
        model=model,
        optimizer=optimizer,
        dataloader=loader.train(),
        lr_scheduler=lr_schedule,
        device=device,
        storage=storage,
        logger=logger
    )

    return task


def main(**kwargs):
    args = Namespace(**kwargs)
    set_verbose_level(args.verbose)

    device = fetch_device()
    experiment_name = args.experiment_name.format(**kwargs)

    client = TrackLogger(experiment_name, storage_uri=args.database)

    # save partial results here
    state_storage = StateStorage(folder=option('state.storage', f'{base}/a2c'), time_buffer=30)

    def main_task():
        return a2c_baseline(device=device, logger=client, storage=state_storage, **kwargs)

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

    hpo.fit(epochs=fidelity(args.epochs), objective='loss')

    # Train using train+valid for the final result
    final_task = a2c_baseline(device=device, logger=client, storage=state_storage, **kwargs)

    if option('worker.id', 0, type=int) == 0:
        print('Waiting for other workers to finish')
        hpo.wait_done()
        if hpo.is_broken():
            return

        params = hpo.best_trial.params
        task_args = params.pop('task')

        final_task.init(**params)
        final_task.fit(**task_args)

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
