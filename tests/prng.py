import torch

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler
from mlbaselines.sampler import RandomSampler
from torch.utils.data.dataset import TensorDataset


data = torch.ones((100, 1))

for i in range(100):
    data[i] = data[i] * i


dataset = TensorDataset(
    data
)


sampler = RandomSampler(
    dataset,
    seed=1
)

batch_sampler = BatchSampler(
    sampler,
    batch_size=2,
    drop_last=True
)

loader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    num_workers=2,
    batch_size=1
)


for b in loader:
    print(b[0])


