from argparse import ArgumentParser

from olympus.datasets import DataLoader, Dataset, SplitDataset
from olympus.models import Model
from olympus.optimizers import Optimizer
from olympus.optimizers.schedules import LRSchedule
from olympus.tasks import ObjectDetection
from olympus.utils.storage import StateStorage
from olympus.utils import fetch_device, option


parser = ArgumentParser()
parser.add_argument('--epochs', type=int, default=3)
args = parser.parse_args()
device = fetch_device()
base = option('base_path', '/tmp/olympus')


def reduce_loss(loss_dict):
    return sum(loss for loss in loss_dict.values())


def make_detection_task(client=None):
    dataset = SplitDataset(
        Dataset('test_pennfudan', path=f'{base}/data'),
        split_method='original'
    )

    loader = DataLoader(
        dataset,
        sampler_seed=0,
        batch_size=1
    )

    input_size, target_size = loader.get_shapes()

    model = Model(
        'fasterrcnn_resnet18_fpn',
        input_size=input_size,
        output_size=dataset.dataset.dataset.num_classes,
        weight_init='glorot_uniform'
    )

    optimizer = Optimizer(
        'sgd',
        lr=0.01,
        momentum=0.99,
        weight_decay=1e-3
    )

    lr_schedule = LRSchedule(
        'exponential',
        gamma=0.97
    )

    main_task = ObjectDetection(
        detector=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedule,
        dataloader=loader.train(),
        device=device,
        criterion=reduce_loss,
        storage=StateStorage(folder=f'{base}/detection_short', time_buffer=0),
        logger=client)

    return main_task


detection = make_detection_task()

# Register the trial
detection.init()

detection.fit(epochs=args.epochs)
detection.report()
