import torch.nn.functional as F
from olympus.datasets import DataLoader

from olympus.optimizers.schedules import LRSchedule
from olympus.optimizers import Optimizer
from olympus.models import Model
from olympus.metrics import MetricList, ProgressView
from olympus.utils import fetch_device


epochs = 2
device = fetch_device()

# Model
model = Model(
    'resnet18',
    input_size=(1, 28, 28),
    output_size=(10,),
    weight_init='glorot_uniform',
    seed=1
)

# Optimizer
optimizer = Optimizer('sgd', params=model.parameters(), weight_decay=0.001, lr=1e-5, momentum=1e-5)

# Schedule
lr_schedule = LRSchedule('exponential', optimizer=optimizer, gamma=0.99)

# Dataloader
loader = DataLoader(
    'test-mnist',
    seed=1,
    sampling_method={'name': 'original'},
    batch_size=32
)

# event handler
event_handler = MetricList()
event_handler.append(
    ProgressView(max_epoch=epochs, max_step=len(loader.train())).every(epoch=1, batch=1))


model = model.to(device=device)
for e in range(epochs):
    losses = []

    for step, (batch, target) in enumerate(loader.train()):

        optimizer.zero_grad()
        predict = model(batch.to(device=device))

        loss = F.cross_entropy(predict, target.to(device=device))
        losses.append(loss.detach())

        optimizer.backward(loss)
        optimizer.step()

        event_handler.on_new_batch(step)

    event_handler.on_new_epoch(e + 1)
    losses = [l.item() for l in losses]
    loss = sum(losses) / len(losses)

event_handler.finish()
