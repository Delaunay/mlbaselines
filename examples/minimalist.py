import torch.nn.functional as F
from olympus.datasets import DataLoader

from olympus.optimizers.schedules import LRSchedule
from olympus.optimizers import Optimizer
from olympus.models import Model
from olympus.metrics import MetricList, ProgressView

# Model
model = Model(
    'resnet18',
    input_size=(1, 28, 28),
    output_size=(10,),
    weight_init='glorot_uniform',
    seed=1
)

# Optimizer
optimizer = Optimizer('sgd', model.parameters(), weight_decay=0.001, lr=1e-5, momentum=1e-5)

# Schedule
lr_schedule = LRSchedule('exponential', optimizer, gamma=0.99)

# Dataloader
loader = DataLoader(
    'mnist',
    seed=1,
    sampling_method={'name': 'original'},
    batch_size=128
)

# event handler
event_handler = MetricList()
event_handler.append(
    ProgressView(max_epoch=5, max_step=len(loader.train())).every(epoch=1, batch=1))

for e in range(5):
    losses = []

    for step, (batch, target) in enumerate(loader.train()):

        optimizer.zero_grad()
        predict = model(batch.cuda())

        loss = F.cross_entropy(predict, target.cuda())
        losses.append(loss.detach())

        optimizer.backward(loss)
        optimizer.step()

        event_handler.step(step)

    event_handler.epoch(e + 1)
    losses = [l.item() for l in losses]
    loss = sum(losses) / len(losses)

event_handler.finish()
