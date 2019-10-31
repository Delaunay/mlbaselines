import torch.distributed
import torch.nn
import sys

from olympus.utils.options import options

_stdout = None

# From 0 to N representing each process running in parallel
# a _rank == -1 means the process is the launcher process or a single GPU training process
_rank = -1


def rank():
    return _rank


def set_rank(rank_):
    global _rank
    _rank = rank_


class NoOut:
    def __init__(self):
        pass

    def write(self, string):
        pass

    def flush(self):
        pass


def arguments(parser):
    parser.add_argument('--rank', type=int, default=0,
                        help='current process rank')
    parser.add_argument('--world-size', type=int, default=1,
                        help='Number of process running in parallel')
    parser.add_argument('--dist-url', type=str, default='nccl:tcp://localhost:8181',
                        help='distributed backend string')

    return parser


def enable_distributed_process(rank, dist_url, world_size,
                               silence_stdout=options('distributed.noprint', True)):
    global _stdout

    if rank is None:
        return

    if world_size > 1:
        set_rank(rank)

        backend, url = dist_url.split(':', maxsplit=1)
        torch.distributed.init_process_group(
            backend=backend,
            init_method=url,
            rank=rank,
            world_size=world_size
        )

        if rank != 0 and silence_stdout:
            _stdout = sys.stdout
            sys.stdout = NoOut()


def data_parallel(model, device_ids=None, *args, **kwargs):
    if device_ids is not None or rank() != -1:
        return torch.nn.parallel.DistributedDataParallel(model, device_ids, *args, **kwargs)
    return model
