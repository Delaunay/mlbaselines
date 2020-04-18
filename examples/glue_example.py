import pprint

import torch

from olympus.baselines.classification import classification_baseline
from olympus.utils.storage import StateStorage
from olympus.utils.options import option, set_option

set_option('progress.frequeuency.epoch', 0)
set_option('progress.frequeuency.batch', 0)

# For checkpoints
storage = StateStorage(folder=option('state.storage', './checkpoints'), time_buffer=5 * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
task = classification_baseline(
    "bert-rte", 'glorot_uniform', 'adam', 'none', "glue-rte", 1, device,
    storage=storage, half=False)


task.init(
    model={'initializer': {'gain': 1.0}},
    optimizer={
        'lr': 0.0001,
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 0.01},
    lr_schedule={})

task.fit(epochs=4)

pprint.pprint(task.metrics.value())
