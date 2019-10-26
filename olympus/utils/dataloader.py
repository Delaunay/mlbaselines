from torch.utils.data import DataLoader as TorchLoader


class GenericStateIterator:
    """Generic Iterator that accepts arguments for every `next` call"""

    def __init__(self, loader):
        self.loader = loader

    def next(self, *args, **kwargs):
        raise NotImplementedError()


class MLIterator(GenericStateIterator):
    """Adapt a Python Iterator into a MLbaseline Iterator for consistency with RLIterator"""
    def __init__(self, loader, python_iterator):
        super(MLIterator, self).__init__(loader)

        self.iterator = python_iterator

    def next(self, *args, **kwargs):
        try:
            self.loader.batch_id += 1
            return next(self.iterator)

        except StopIteration:
            self.loader.batch_id = 0
            self.loader.epoch_id += 1
            return None


class DataLoader:
    def __init__(self):
        self.batch_id = 0
        self.epoch_id = 0

    def iterator(self):
        raise NotImplementedError()


class TorchDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(TorchDataloader, self).__init__()

        self.loader = TorchLoader(*args, **kwargs)

    def __len__(self):
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset

    def iterator(self):
        return MLIterator(self, iter(self.loader))
