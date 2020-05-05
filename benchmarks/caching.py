import argparse
import time

import torch.cuda
from torch.nn import CrossEntropyLoss

from olympus.datasets import Dataset, DataLoader, SplitDataset
from olympus.models import Model
from olympus.optimizers import Optimizer
from olympus.utils.stat import StatStream
from olympus.utils import show_dict, fetch_device


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--model', default='vgg11', type=str)
parser.add_argument('--caching', action='store_true', dest='caching')
parser.add_argument('--no-caching', action='store_false', dest='caching')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--warmup', default=4, type=int)
parser.add_argument('--repeat', default=10, type=int)
args = parser.parse_args()
show_dict(vars(args))

device = fetch_device()
if args.caching:
    args.caching = device

dataset = SplitDataset(
    Dataset(
        args.dataset,
        cache=args.caching,
        transform=False),
    split_method='original'
)
loaders = DataLoader(dataset, batch_size=args.batch_size, sampler_seed=0)
input_size, target_size = loaders.get_shapes()

model = Model(
    args.model,
    input_size=input_size,
    output_size=target_size[0]
).init()

optimizer = Optimizer(
    'sgd',
    params=model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.001)

criterion = CrossEntropyLoss()
model = model.to(device=device)
train_loader = loaders.train()


def epoch():
    for batch in train_loader:
        x, *_, y = batch

        output = model(x[0].to(device=device))
        loss = criterion(output, y.to(device=device))

        optimizer.backward(loss)
        optimizer.step()


print('Warmup')
for i in range(args.warmup):
    print(f'\r{i + 1:3d}/{4:3d}', end='')
    epoch()

print('\nBenchmark')
stats = StatStream(drop_first_obs=1)
for i in range(args.repeat):
    print(f'\r{i + 1:3d}/{10:3d}', end='')
    start = time.time()

    epoch()
    torch.cuda.synchronize()
    stats.update(time.time() - start)

print()
show_dict(stats.to_json())
