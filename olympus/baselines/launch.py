from argparse import ArgumentParser, REMAINDER
import os
import json
import signal
import subprocess
import sys

from olympus.utils.gpu import GpuMonitor
from olympus.utils import info


def get_available_tasks():
    __ignored_files = {
        '__init__.py',
        '__pycache__',
        'launch.py'
    }

    scripts = []

    for file_name in os.listdir(os.path.dirname(__file__)):
        if file_name not in __ignored_files:
            scripts.append(file_name.replace('.py', ''))

    return scripts


def arguments():
    parser = ArgumentParser(description="Olympus task launcher")
    parser.add_argument('--no-mon', action='store_true', default=False,
                        help='Disable GPU monitoring')

    # When training on multi GPUs you can specify the devices on which to train
    parser.add_argument('--devices', type=str, default='0',
                        help='GPU ids used for the training (comma separated value)')

    parser.add_argument('task', choices=get_available_tasks())

    # rest from the training program
    parser.add_argument('args', nargs=REMAINDER)
    return parser


def get_available_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def multigpu_launch(task_name, script_args, job_env, device_id, rank, world_size, port):
    """Launch the task using multiple GPUs"""
    info(f'Launching job on (device: {device_id})')

    script = f'{os.path.dirname(__file__)}/{task_name}.py'

    cmd = list([f'CUDA_VISIBLE_DEVICES={device_id}', sys.executable, '-u'])
    cmd.append(script)
    cmd.extend(('--rank', str(rank)))
    cmd.extend(('--world-size', str(world_size)))
    cmd.extend(('--dist-url', f'nccl:tcp://localhost:{port}'))
    cmd.extend(script_args)

    return subprocess.Popen(' '.join(cmd), env=job_env, shell=True)


def simple_launch(task_name, script_args, job_env, device_id, rank, world_size, port):
    """Launch the task without creating another python interpreter"""
    module = __import__("olympus.baselines.{}".format(task_name), fromlist=[''])
    parser = module.arguments()
    args = parser.parse_args(script_args)
    module.main(**vars(args))
    return None


def single_gpu_launch(task_name, script_args, job_env, device_id, rank, world_size, port):
    """Launch the task for a given GPU"""
    info(f'Launching job on (device: {device_id})')

    script = f'{os.path.dirname(__file__)}/{task_name}.py'

    cmd = list([f'CUDA_VISIBLE_DEVICES={device_id}', sys.executable, '-u'])
    cmd.append(script)
    cmd.extend(script_args)

    return subprocess.Popen(' '.join(cmd), env=job_env, shell=True)


def main(argv=None):
    args = arguments().parse_args(argv)
    args.devices = [int(d) for d in args.devices.split(',')]
    job_env = os.environ
    processes = []
    world_size = len(args.devices)
    launcher = simple_launch
    port = None
    script_args = args.args

    # Pytorch by default uses GPU0 so we just keep the simple launch
    # but if the user specifies a specific GPU we need to set CUDA_VISIBLE_DEVICES
    if args.devices != [0]:
        launcher = single_gpu_launch

    if world_size > 1:
        launcher = multigpu_launch
        port = get_available_port()

    with GpuMonitor(enabled=not args.no_mon) as mon:
        try:
            for rank, device_id in enumerate(args.devices):
                process = launcher(
                    args.task, script_args, job_env, device_id, rank, world_size, port)

                if process:
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
