import os
import sys

from olympus.utils import parse_uri, info
from olympus.utils.network import get_network_interface, get_ip_address, get_free_port


def make_cockroach_server(uri, *args, **kwargs):
    from olympus.distributed.cockroachdb import CockRoachDB
    data = parse_uri(uri)

    return CockRoachDB(
        location='/tmp/cockroach/queue',
        addrs=f'{data.get("address")}:{data.get("port")}'
    )


def make_cockroach_client(uri, *args, **kwargs):
    from olympus.distributed.cockroachdb import CKMQClient
    return CKMQClient(uri, *args, **kwargs)


def make_python_client(uri, *args, **kwargs):
    from olympus.distributed.pyqueue import PythonQueue
    return PythonQueue()


def make_python_broker(uri, *args, **kwargs):
    from olympus.distributed.pyqueue import PythonBroker
    return PythonBroker()


def make_mongo_client(uri, *args, **kwargs):
    from olympus.distributed.mongo import MongoClient
    return MongoClient(uri)


def make_mongo_broker(uri, *args, **kwargs):
    from olympus.distributed.mongo import MongoDB
    options = parse_uri(uri)
    return MongoDB(options['address'], port=int(options['port']), location='/tmp/mongo')


client_factory = {
    'cockroach': make_cockroach_client,
    'python': make_python_client,
    'mongodb': make_mongo_client,
}

broker_factory = {
    'cockroach': make_cockroach_server,
    'python': make_python_broker,
    'mongodb': make_mongo_broker
}


def make_message_broker(uri, *args, **kwargs):
    options = parse_uri(uri)
    return broker_factory.get(options.get('scheme'))(uri, *args, **kwargs)


def make_message_client(uri, *args, **kwargs):
    options = parse_uri(uri)
    return client_factory.get(options.get('scheme'))(uri, *args, **kwargs)


def get_main_script():
    import inspect
    stack = inspect.stack()
    return stack[-1].filename


def launch(node, cmd, environ):
    cmd = f'ssh -q {node} nohup {cmd} > {node}.out 2> {node}.err < /dev/null &'
    info(cmd)
    return True


def init_process_group(backend, world_size=None, rank=None, node_list=None, addr=None, port=None):
    # https://www.glue.umd.edu/hpcc/help/software/pytorch.html

    def _select(a, b):
        if a is None:
            return b
        return a

    init = not bool(os.getenv('MLBASELINES_WORKER', False))

    addr        = _select(addr      , os.environ.get('MLBASELINES_ADDR'))
    port        = _select(port      , os.environ.get('MLBASELINES_PORT'))
    world_size  = _select(world_size, int(os.environ.get('SLURM_NPROCS', 1)))
    rank        = _select(rank      , int(os.environ.get('SLURM_PROCID', 1)))
    node_list   = _select(node_list , os.environ.get('SLURM_JOB_NODELIST'))

    if node_list is None:
        node_list = ['localhost']

    if addr is None or port is None:
        addr = get_ip_address(get_network_interface())
        port = get_free_port()

    init_method = f'{backend}://{addr}:{port}'
    info(f'initializing using {init_method}')

    if init:
        from mlbaselines.prime import PrimeMonitor
        init_node = PrimeMonitor(init_method, world_size=world_size)

        environ = dict(
            MLBASELINES_WORKER='1',
            MLBASELINES_ADDR=addr,
            MLBASELINES_PORT=port
        )

        script = _select(os.getenv('MLBASELINES_SCRIPT'), get_main_script())
        argv = ' '.join(sys.argv)

        info(f'Using {script} and {argv}')

        # for each node spawn a worker
        for node in node_list:
            launch(node, f'{sys.executable} {script} {argv}', environ)

        # check and exit

        # make the HPO
        init_node.queue_hpo()

        # make a redundant broker
        init_node.queue_broker()
        init_node.queue_broker()

        init_node.shutdown()
    else:
        from olympus.worker import Worker
        worker = Worker(init_method, worker_id=f'worker-{rank}')
        worker.run()

    sys.exit(0)


# nohup myprogram > foo.out 2> foo.err < /dev/null &
