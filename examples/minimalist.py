import torch.nn.functional as F
from olympus.datasets import Dataset, SplitDataset, DataLoader
from olympus.optimizers import Optimizer, LRSchedule

from olympus.models import Model
from olympus.observers import ObserverList, ProgressView
from olympus.utils import fetch_device, option


epochs = 2
device = fetch_device()
base = option('base_path', '/tmp/olympus')

# Model
model = Model(
    'resnet18',
    input_size=(1, 28, 28),
    output_size=(10,)
)

# Optimizer
optimizer = Optimizer('sgd', params=model.parameters(), weight_decay=0.001, lr=1e-5, momentum=1e-5)

# Schedule
lr_schedule = LRSchedule('exponential', optimizer=optimizer, gamma=0.99)

data = Dataset('fake_mnist', path=f'{base}/data')

splits = SplitDataset(data, split_method='original')

# Dataloader
loader = DataLoader(
    splits,
    sampler_seed=1,
    batch_size=32
)

# event handler
event_handler = ObserverList()
event_handler.append(
    ProgressView(max_epoch=epochs, max_step=len(loader.train())).every(epoch=1, batch=1))

model = model.to(device=device)
loss = 0

event_handler.start_train()

for e in range(epochs):
    losses = []
    event_handler.new_epoch(e + 1)

    for step, ((batch, ), target) in enumerate(loader.train()):
        event_handler.new_batch(step)

        optimizer.zero_grad()
        predict = model(batch.to(device=device))

        loss = F.cross_entropy(predict, target.to(device=device))
        losses.append(loss.detach())

        optimizer.backward(loss)
        optimizer.step()

        event_handler.end_batch(step)

    losses = [l.item() for l in losses]
    loss = sum(losses) / len(losses)
    event_handler.end_epoch(e + 1)

event_handler.end_train()
print(loss)
