from argparse import ArgumentParser, REMAINDER
import os
import signal
import subprocess
import sys


__ignored_files = {
    '__init__.py',
    'launch.py'
}


def get_available_tasks():
    scripts = []

    for file_name in os.listdir(os.path.dirname(__file__)):
        if file_name not in __ignored_files:
            scripts.append(file_name.replace('.py', ''))

    return scripts


__tasks = get_available_tasks()


def parse_args():
    parser = ArgumentParser(description="Olympus task launcher")
    # When training on multi GPUs you can specify the devices on which to train
    parser.add_argument('--devices', type=int, nargs='*', default=(0,),
                        help='GPU ids used for the training')

    # Optional arguments for the launch helper
    parser.add_argument("--name", type=str, choices=__tasks,
                        help="task to launch")

    # rest from the training program
    parser.add_argument('args', nargs=REMAINDER)
    return parser.parse_args()


def get_available_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def multigpu_launch(script, args, job_env, device_id, rank, world_size, port):
    cmd = [f'CUDA_VISIBLE_DEVICES={device_id}', sys.executable, '-u']
    cmd.append(script)
    cmd.extend(('--rank', str(rank)))
    cmd.extend(('--world-size', str(world_size)))
    cmd.extend(('--dist-url', f'nccl:tcp://localhost:{port}'))
    cmd.extend(args)

    return subprocess.Popen(' '.join(cmd), env=job_env, shell=True)


def simple_launch(script, args, job_env, device_id, rank, world_size, port):
    cmd = [f'CUDA_VISIBLE_DEVICES={device_id}', sys.executable, "-u"]
    cmd.append(script)
    cmd.extend(args)

    return subprocess.Popen(' '.join(cmd), env=job_env, shell=True)


def main():
    args = parse_args()
    script = f'{os.path.dirname(__file__)}/{args.name}.py'

    job_env = os.environ
    processes = []
    world_size = len(args.devices)
    launcher = simple_launch
    port = None

    if world_size > 1:
        launcher = multigpu_launch
        port = get_available_port()

    try:
        for rank, device_id in enumerate(args.devices):
            process = launcher(
                script,    args.args, job_env,
                device_id, rank,      world_size, port)
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
