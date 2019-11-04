import torch.nn.functional as F
from olympus.datasets import build_loaders

from olympus.optimizers.schedules import LRSchedule
from olympus.optimizers import Optimizer
from olympus.models import Model

# Model
model = Model(
    'resnet18',
    input_size=(1, 28, 28),
    output_size=(10,)
).cuda()

# Optimizer
optimizer = Optimizer(
    'sgd'
).init_optimizer(
    model.parameters(),
    weight_decay=0.001,
    lr=1e-5,
    momentum=1e-5
)

# Schedule
lr_schedule = LRSchedule(
    'exponential'
).init_schedule(
    optimizer,
    gamma=0.99
)

# Dataloader
datasets, loaders = build_loaders(
    'mnist',
    seed=0,
    sampling_method={'name': 'original'},
    batch_size=128
)


train_loader = loaders['train']

for e in range(5):
    losses = []

    for step, (batch, target) in enumerate(train_loader):

        optimizer.zero_grad()
        predict = model(batch.cuda())

        loss = F.cross_entropy(predict, target.cuda())
        losses.append(loss.detach())

        optimizer.backward(loss)
        optimizer.step()
        print(f'\r[{e:3d}] [{step:3d}] {loss:.4f}', end='')

    print()
    losses = [l.item() for l in losses]
    loss = sum(losses) / len(losses)

    print(f'[{e:3d}] {loss:.4f}')
