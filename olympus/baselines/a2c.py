from argparse import ArgumentParser, Namespace

from olympus.models import Model, known_models
from olympus.models.inits import known_initialization, Initializer

from olympus.reinforcement import Environment
from olympus.reinforcement.dataloader import RLDataLoader, simple_replay_vector
from olympus.observers import ElapsedRealTime
from olympus.optimizers import Optimizer, known_optimizers
from olympus.optimizers.schedules import LRSchedule, known_schedule

from olympus.tasks.reinforcement.a2c import A2C
from olympus.hpo import HPOptimizer, Fidelity
from olympus.tasks.hpo import HPO

from olympus.utils import fetch_device, set_verbose_level, show_dict, required
from olympus.utils.functional import flatten
from olympus.utils.options import option
from olympus.utils.storage import StateStorage

DEFAULT_EXP_NAME = 'reinforcement_{env_name}_{model}_{optimizer}_{lr_scheduler}_{weight_init}'
base = option('base_path', '/tmp/olympus')


def arguments():
    parser = ArgumentParser(prog='a2c', description='a2c Baseline')

    parser.add_argument(
        '--experiment-name', type=str, default=DEFAULT_EXP_NAME, metavar='EXP_NAME',
        help='Name of the experiment in Orion storage (default: {})'.format(DEFAULT_EXP_NAME))
    parser.add_argument(
        '--epochs', type=int, default=300, metavar='N',
        help='maximum number of epochs to train (default: 300)')
    parser.add_argument(
        '--min-epochs', type=int, default=20, metavar='MN',
        help='minimum number of epochs to train (default: 20) for HPO')

    parser.add_argument(
        '--model', type=str, metavar='MODEL_NAME', choices=known_models(), default=required,
        help='Name of the model')
    parser.add_argument(
        '--weight-init', type=str, default='glorot_uniform', metavar='INIT_NAME',
        choices=known_initialization(),
        help='Name of the initialization (default: glorot_uniform)')
    parser.add_argument(
        '--model-seed', type=int, default=1,
        help='random seed for model initialization (default: 1)')

    parser.add_argument(
        '--env-name', type=str, default=required,
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
                 lr_scheduler, num_steps, half, device, storage, **config):
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
        storage=storage
    )

    return task


def main(**kwargs):
    args = Namespace(**kwargs)
    set_verbose_level(args.verbose)

    device = fetch_device()
    experiment_name = args.experiment_name.format(**kwargs)

    # save partial results here
    state_storage = StateStorage(folder=option('state.storage', f'{base}/a2c'))

    def main_task():
        return a2c_baseline(device=device, storage=state_storage, **kwargs)

    space = main_task().get_space()

    # If space is not empty we search the best hyper parameters
    params = {}
    if space:
        show_dict(space)
        hpo = HPOptimizer('hyperband', space=space,
                          fidelity=Fidelity(args.min_epochs, args.epochs).to_dict())

        hpo_task = HPO(hpo, main_task)
        hpo_task.metrics.append(ElapsedRealTime())

        trial = hpo_task.fit(objective='loss')
        print(f'HPO is done, objective: {trial.objective}')
        params = trial.params
    else:
        print('No hyper parameter missing, running the experiment...')
    # ------

    # Run the experiment with the best hyper parameters
    # -------------------------------------------------
    if params is not None:
        # Train using train + valid for the final result
        final_task = a2c_baseline(device=device, storage=state_storage, **kwargs, hpo_done=True)
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
