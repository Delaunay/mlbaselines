from argparse import ArgumentParser, REMAINDER
from dataclasses import dataclass, field
import os
import json
import signal
import subprocess
import sys
from typing import List, Dict

from olympus.utils.gpu import GpuMonitor
from olympus.utils import info, warning, parse_args


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
    parser.add_argument('--devices', type=str, default=None,
                        help='GPU ids used for the training (comma separated value)')

    # You can have more than one worker
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of workers')

    parser.add_argument('--device-sharing', action='store_true', default=True, dest='device_sharing',
                        help='Enable device sharing across workers')

    parser.add_argument('--no-device-sharing', action='store_false', dest='device_sharing',
                        help='Enable device sharing across workers')

    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Workers do not need devices')

    # parser.add_argument('--rdv-url', type=str, default=None,
    #                     help='URL used to sync multi node workers for multi node training')

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


def local_multigpu_launch(task_name, script_args, job_env, device_id, rank, world_size, port):
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


def simple_launch(task_name, script_args):
    """Launch the task without creating another python interpreter"""
    module = __import__("olympus.baselines.{}".format(task_name), fromlist=[''])
    parser = module.arguments()

    args = parse_args(parser, script_args)
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


@dataclass
class Worker:
    worker_id: int
    devices: List[int]
    env: Dict = field(default_factory=lambda: os.environ)
    processes: List = field(default_factory=list)

    def launch(self, task_name, script_args):
        """
        Parameters
        ----------
        task_name: str
            name of the task to run

        script_args: List[str]
            list of arguments to pass to the task
        """
        subprocess_factory = single_gpu_launch
        port = None
        self.env['OLYMPUS_WORKER_ID'] = str(self.worker_id)

        if len(self.devices) > 1:
            port = get_available_port()
            subprocess_factory = local_multigpu_launch
            self.env['OLYMPUS_WORKER_PORT'] = str(port)

        self.env['OLYMPUS_WORKER_WORLD'] = str(len(self.devices))
        self.processes = []
        for rank, device_id in enumerate(self.devices):
            proc = subprocess_factory(
                task_name, script_args, self.env, device_id, rank, len(self.devices), port)

            self.processes.append(proc)

        return self.processes


def get_device_count(devices):
    if devices:
        return [int(d) for d in devices.split(',')]

    import torch
    return list(range(torch.cuda.device_count()))


def make_device_groups(worker_count, devices, shared, cpu_mode):
    if not devices or cpu_mode:
        print('Warning no devices detected, It will run in CPU mode')
        devices = [0]

    if shared or cpu_mode:
        return [devices for _ in range(worker_count)]

    device_per_worker = len(devices) // worker_count
    remaining_devices = len(devices) % worker_count

    if device_per_worker == 0:
        raise RuntimeError(
            f'Not enough devices (devices: {len(devices)}) < (workers: {worker_count})'
            'Use --device-sharing or --cpu to bypass this error'
        )

    groups = []
    for wid in range(worker_count):
        groups.append(devices[device_per_worker * wid: device_per_worker * (wid + 1)])

    if remaining_devices > 0:
        warning('Some devices were not assigned to worker')

    return groups


def single_worker_single_gpu(task, args, no_mon):
    with GpuMonitor(enabled=not no_mon) as mon:
        simple_launch(task, args)
        print(json.dumps(mon.to_json(), indent=2))
        show_resource_stats(mon)


def show_resource_stats(monitor):
    return

    import numpy as np
    import pandas as pd
    import altair as alt

    ds = monitor.monitor.ts

    print(ds)
    df = pd.DataFrame(ds)
    print(df)
    df.to_csv('data.csv')
    df['index'] = np.arange(len(ds['utilization.gpu0']))

    chart = (alt.Chart(df)
                .mark_line()
                .encode(x='index', y='utilization.memory0'))

    chart.save('utilization_memory0.png')


def run(workers, all_processes, task, script_args):
    for worker in workers:
        processes = worker.launch(task, script_args)
        all_processes.extend(processes)

    errors = []
    for process in all_processes:
        process.wait()

        if process.returncode != 0:
            errors.append((
                process.returncode,
                process.args
            ))

    for return_code, cmd in errors:
        print(f'Command {cmd} failed with return code {return_code}')


def cleanup(all_processes):
    # propagate the signal to children
    for process in all_processes:
        process.send_signal(signal=signal.SIGINT)

    # wait 5 seconds for them to die
    for process in all_processes:
        try:
            process.wait(timeout=5)

        # kill them if time out
        except subprocess.TimeoutExpired:
            process.terminate()


def main(argv=None):
    args = arguments().parse_args(argv)
    args.devices = get_device_count(args.devices)
    script_args = args.args
    device_groups = make_device_groups(
        args.workers, args.devices, args.device_sharing, args.cpu)

    # If we can do not spawn another python interpreter
    # One GPU or no GPU and a single worker
    if (args.devices == [0] or args.devices == []) and args.workers <= 1:
        single_worker_single_gpu(args.task, script_args, args.no_mon)
        return

    # We have to spawn interpreters
    # Create work groups
    workers = []
    for wid, devices in zip(range(args.workers), device_groups):
        worker = Worker(wid, devices=devices)
        workers.append(worker)

    all_processes = []
    with GpuMonitor(enabled=not args.no_mon) as mon:
        try:
            run(workers, all_processes, args.task, script_args)
            show_resource_stats(mon)
            print(json.dumps(mon.to_json(), indent=2))

        except KeyboardInterrupt:
            cleanup(all_processes)


if __name__ == '__main__':
    main()
