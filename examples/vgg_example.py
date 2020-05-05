import pprint

import torch

from olympus.baselines.classification import classification_baseline
from olympus.utils.storage import StateStorage
from olympus.utils.options import set_option

set_option('progress.frequeuency.epoch', 1)
set_option('progress.frequeuency.batch', 1)
set_option('progress.show.metrics', 'epoch')

# For checkpoints
storage = StateStorage(folder='./checkpoints', time_buffer=5 * 60)

task = classification_baseline(
    "vgg11", 'glorot_uniform', 'sgd', 'exponential', "cifar10", 128, torch.device('cuda'),
    storage=storage, half=False, transform=False, cache=torch.device('cuda'))


task.init(
    model={'initializer': {'gain': 1.0}},
    optimizer={
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.001},
    lr_schedule={'gamma': 0.98})

task.fit(epochs=120)

task.finish()

pprint.pprint(task.metrics.value())
