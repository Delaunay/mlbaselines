from argparse import ArgumentParser

from olympus.datasets import DataLoader
from olympus.models import Model
from olympus.optimizers import Optimizer
from olympus.optimizers.schedules import LRSchedule
from olympus.tasks import ObjectDetection
from olympus.utils.storage import StateStorage
from olympus.utils import fetch_device


parser = ArgumentParser()
parser.add_argument('--epochs', type=int, default=3)
args = parser.parse_args()
device = fetch_device()


def reduce_loss(loss_dict):
    return sum(loss for loss in loss_dict.values())


def make_detection_task(client=None):
    dataset = DataLoader(
        'pennfudan',
        seed=0,
        sampling_method={'name': 'original'},
        batch_size=1)

    input_size, target_size = dataset.get_shapes()

    model = Model(
        'fasterrcnn_resnet18_fpn',
        input_size=input_size,
        output_size=dataset.datasets.num_classes,
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
        dataloader=dataset.train(),
        device=device,
        criterion=reduce_loss,
        storage=StateStorage(folder='/tmp/olympus/detection', time_buffer=0),
        logger=client)

    return main_task


detection = make_detection_task()

# Register the trial
detection.init()

detection.fit(epochs=args.epochs)
detection.report()
