from torch import Tensor
from torch.utils.data import Dataset

from olympus.utils import warning


def _try_to(maybe_tensor, device):
    if isinstance(maybe_tensor, Tensor):
        return maybe_tensor.to(device)
    else:
        return maybe_tensor


class DatasetCache(Dataset):
    """Caches samples, as they are loaded from the specified dataset, to the specified device

    Notes
    -----

    While caching might seem like a simple way to speed-up data loading it is important to know
    your pipeline.

    This class will cache the data right before the collation of images by the dataloader.
    It means that you cannot use randomized transforms as those will get cached and not called again.

    1. Reading from disk
    2. Decoding (JPEG -> Pixel Array)
    3. Dataset Transforms
    4. **Caching**
    5. Collate images into a batch of images

    Additionally, the linux kernel already cache recently opened files in memory.
    Caching should show no improvement for small dataset like CIFAR or MNIST.

    Caching can be useful when using big batches on a small network, as it will reduce IO operations,
    For example using PascalVOC with resnet18 with the caching can show 15% decrease in epoch times..

    """
    def __init__(self, dataset, device):
        """
        Args:
            dataset (torch.utils.data.Dataset): dataset to cache
            device  (torch.device): device where cached samples will be stored
        """
        warning('DatasetCache must only be used with small datasets')

        if device.type == 'cuda':
            warning('Warning: pin_memory must be set to \'False\' when caching to a cuda device')

        self.dataset = dataset
        self.device = device
        self.cache = {}

    def __getitem__(self, idx):
        if idx not in self.cache:
            sample = self.dataset[idx]
            self.cache[idx] = tuple(_try_to(data, self.device) for data in sample)
        return self.cache[idx]

    def __len__(self):
        return len(self.dataset)
