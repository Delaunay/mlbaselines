from argparse import ArgumentParser, REMAINDER
import os
import json
import signal
import subprocess
import sys

from olympus.utils.gpu import GpuMonitor
from olympus.utils.factory import fetch_factories


__ignored_files = {
    '__init__.py',
    '__pycache__',
    'launch.py'
}


def get_available_tasks():
    scripts = []

    for file_name in os.listdir(os.path.dirname(__file__)):
        if file_name not in __ignored_files:
            scripts.append(file_name.replace('.py', ''))

    return scripts


task_factories = fetch_factories('olympus.scripts', __file__, function_name='arg_parser')


def add_parsers(parser):
    subparsers = parser.add_subparsers(dest='task', help='Tasks')
    for task_name, task_factory in task_factories.items():
        task_factory(subparsers)


def main_arg_parser():
    parser = ArgumentParser(description="Olympus task launcher")
    parser.add_argument('--no-mon', action='store_true', default=False,
                        help='Disable GPU monitoring')

    # When training on multi GPUs you can specify the devices on which to train
    parser.add_argument('--devices', type=int, nargs='*', default=(0,),
                        help='GPU ids used for the training')

    add_parsers(parser)

    # rest from the training program
    # parser.add_argument('args', nargs=REMAINDER)

    return parser


def get_available_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def multigpu_launch(task_name, args, job_env, device_id, rank, world_size, port):
    script = f'{os.path.dirname(__file__)}/{args.task}.py'

    cmd = [f'CUDA_VISIBLE_DEVICES={device_id}', sys.executable, '-u']
    cmd.append(script)
    cmd.extend(('--rank', str(rank)))
    cmd.extend(('--world-size', str(world_size)))
    cmd.extend(('--dist-url', f'nccl:tcp://localhost:{port}'))
    cmd.extend(args)

    return subprocess.Popen(' '.join(cmd), env=job_env, shell=True)


def debug_launch(task_name, args, job_env, device_id, rank, world_size, port):
    module = __import__("olympus.scripts.{}".format(task_name), fromlist=[''])

    module.main(**args)
    
    return subprocess.Popen('echo')  # Do nothing...

 
def simple_launch(task_name, args, job_env, device_id, rank, world_size, port):
    script = f'{os.path.dirname(__file__)}/{args.task}.py'

    cmd = [f'CUDA_VISIBLE_DEVICES={device_id}', sys.executable, "-u"]
    cmd.append(script)
    cmd.extend(args)

    return subprocess.Popen(' '.join(cmd), env=job_env, shell=True)


def main(argv=None):
    args = main_arg_parser().parse_args(argv)
    job_env = os.environ
    processes = []
    world_size = len(args.devices)
    launcher = debug_launch
    port = None

    if world_size > 1:
        launcher = multigpu_launch
        port = get_available_port()

    with GpuMonitor(enabled=not args.no_mon) as mon:
        try:
            for rank, device_id in enumerate(args.devices):
                process = launcher(
                    args.task, vars(args), job_env, device_id, rank, world_size, port)
                processes.append(process)

            errors = []
            for process in processes:
                process.wait()

                if process.returncode != 0:
                    errors.append((
                        process.returncode,
                        process.args
                    ))

            for return_code, cmd in errors:
                print(f'Command {cmd} failed with return code {return_code}')

            print(json.dumps(mon.to_json(), indent=2))

        except KeyboardInterrupt:
            # propagate the signal to children
            for process in processes:
                process.send_signal(signal=signal.CTRL_C_EVENT)

            # wait 5 seconds for them to die
            for process in processes:
                try:
                    process.wait(timeout=5)

                # kill them if time out
                except subprocess.TimeoutExpired:
                    process.terminate()


if __name__ == '__main__':
    main()
