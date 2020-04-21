import functools
from dataclasses import dataclass, field
from typing import List

import torch
from olympus.datasets.dataset import AllDataset


@dataclass
class Multivariate:
    mean: List = field(default_factory=list)
    sd: List = field(default_factory=list)

    def __len__(self):
        return len(self.mean)


class Gaussian(AllDataset):
    def __init__(self, size, distributions):
        self.size = size
        self.distribution = distributions
        self.num_classes = len(distributions)
        self.features_size = len(distributions[0])
        self.data = None
        self.generate()
        super(Gaussian, self).__init__(
            self,
            input_shape=(self.features_size,),
            target_shape=(self.features_size,),
            train_size=int(len(self) * 0.8),
            valid_size=int(len(self) * 0.1),
            test_size=int(len(self) * 0.1)
        )

    def generate(self):
        X = torch.zeros(self.size * self.num_classes, self.features_size, dtype=torch.float)
        Y = torch.zeros(self.size * self.num_classes, dtype=torch.float)

        for cls, gauss in enumerate(self.distribution):
            start = cls * self.size
            end = (cls + 1) * self.size

            for i, (m, s) in enumerate(zip(gauss.mean, gauss.sd)):
                dat = torch.normal(m, s, (self.size,))
                X[start:end, i] = dat

            Y[start:end] = cls

        self.data = (X, Y)

    def __getitem__(self, index):
        X, Y = self.data
        return dict(batch=X[index], target=Y[index])

    def __len__(self):
        return self.size * self.num_classes


builders = {
    'gaussian': Gaussian,
    'gaussian-2': functools.partial(Gaussian, distributions=[
        Multivariate([1, 2], [2, 1]),
        Multivariate([2, 1], [1, 2])])
}


