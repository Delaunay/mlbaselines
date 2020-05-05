from torch import Tensor
from torch.utils.data import Dataset


def _try_to(maybe_tensor, device):
    if isinstance(maybe_tensor, Tensor):
        return maybe_tensor.to(device)
    else:
        return maybe_tensor


class DatasetCache(Dataset):
    """
    Caches samples, as they are loaded from the specified dataset, to the specified device
    """
    def __init__(self, dataset, device):
        """
        Args:
            dataset (torch.utils.data.Dataset): dataset to cache
            device  (torch.device): device where cached samples will be stored
        """
        if device.type == 'cuda':
            print('Warning: pin_memory must be set to \'False\' when caching to a cuda device')
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
