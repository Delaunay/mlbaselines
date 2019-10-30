from collections import OrderedDict

import torch
from torch.utils.data.dataset import Dataset


class Gaussian(Dataset):
    def __init__(self, width, n_classes, mean, std, length):
        self.width = width
        self.n_classes = n_classes
        self.mean = mean
        self.std = std
        self.length = length
        self.generate()

    def generate(self):
        mean = self.mean * torch.ones(self.length, self.width)
        std = self.std * torch.ones(self.length, self.width)
        X = torch.normal(mean, std)
        Y = torch.ones(self.length, self.n_classes) * X.sum(1).unsqueeze(1)
        for n in range(self.n_classes):
            Y[:, n] += X[:, n] * X[:, n + 1]
        # Y += Y.min(1)
        # Y /= Y.sum(1)
        Y = Y.argmax(1)
        self.data = (X, Y)

    def __getitem__(self, index):
        return tuple(subdata[index] for subdata in self.data)

    def __len__(self):
        return self.length



def build(batch_size, width, classes, size, data_path):
    size *= width

    loaders = OrderedDict()
    for set_name in ['train', 'valid', 'test']:
        dataset = Gaussian(
            width=width, n_classes=classes,
            mean=0, std=1 / width,
            length=size)

        loaders[set_name] = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size, shuffle=True)


    return loaders
