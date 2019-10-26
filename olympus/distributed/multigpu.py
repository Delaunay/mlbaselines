import torch.distributed
import torch.nn


def arguments(parser):
    parser.add_argument('--rank', type=int, default=0,
                        help='current process rank')
    parser.add_argument('--world-size', type=int, default=1,
                        help='Number of process running in parallel')
    parser.add_argument('--dist-url', type=str, default='nccl:tcp://localhost:8181',
                        help='distributed backend string')

    return parser


def enable_distributed_process(args):
    if args.world_size > 1:
        backend, url = args.dist_url.split(':', maxsplit=1)
        torch.distributed.init_process_group(
            backend=backend,
            init_method=url,
            rank=args.rank,
            world_size=args.world_size
        )


def data_parallel(model, *args, **kwargs):
    return torch.nn.parallel.DistributedDataParallel(model, *args, **kwargs)
