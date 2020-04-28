from torch.utils.data.dataset import Subset


class TransformedSubset(Subset):
    def __init__(self, dataset, indices, transform=None):
        super(TransformedSubset, self).__init__(dataset, indices)

        self.transform = transform

    def __getitem__(self, idx):
        data = super(TransformedSubset, self).__getitem__(idx)
        target = data[-1]
        data = data[:-1]

        if self.transform is not None:
            data = [self.transform(x) for x in data]

        return data, target
