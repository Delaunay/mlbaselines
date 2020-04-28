import argparse
import logging
import pprint

import torch

from olympus.baselines.classification import classification_baseline
from olympus.utils.options import option, set_option
from olympus.utils.storage import StateStorage

set_option('progress.frequeuency.epoch', 0)
set_option('progress.frequeuency.batch', 0)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='either rte or sst2', required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # For checkpoints
    storage = StateStorage(folder=option('state.storage', './checkpoints'), time_buffer=5 * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task = classification_baseline(
        "bert-{}".format(args.task), 'normal', 'adam', 'none', "glue-{}".format(args.task),
        args.batch_size, device, storage=storage, half=args.fp16, hpo_done=True)

    task.init(
        model={'initializer': {'mean': 0.0, 'std': 0.2}},
        optimizer={
            'lr': args.lr,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.01},
        lr_schedule={})

    task.fit(epochs=args.epochs)
    pprint.pprint(task.metrics.value())


if __name__ == '__main__':
    main()
