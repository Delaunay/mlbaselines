from torch.utils.data.dataset import Subset


class TransformedSubset(Subset):
    def __init__(self, dataset, indices, transform=None):
        super(TransformedSubset, self).__init__(dataset, indices)

        self.transform = transform

    def __getitem__(self, idx):
        data, target = super(TransformedSubset, self).__getitem__(idx)

        if self.transform is not None:
            data = self.transform(data)

        return data, target


