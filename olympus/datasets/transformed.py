from torch.utils.data.dataset import Subset


class TransformedSubset(Subset):
    def __init__(self, dataset, indices, transform=None):
        super(TransformedSubset, self).__init__(dataset, indices)

        self.transform = transform

    def __getitem__(self, idx):
        sample = super(TransformedSubset, self).__getitem__(idx)

        batch = sample['batch']
        target = sample['target']

        if self.transform is not None:
            batch = [self.transform(x) for x in batch]

        return {'batch': batch, 'target': target, 0: batch, 1: target}
