from torch.utils.data import Dataset


def identity(x):
    return x


class WindowedDataset(Dataset):
    """Moving window dataset with overlapping observations

    Examples
    --------

    .. code-block:: python

        dataset = StockMarketDataset(['AAPL'], '2000-01-01', '2019-05-10')
        windowed = WindowedDataset(
            dataset,
            window=7,
            transforms=lambda x: x.transpose(1, 0),
            overlaps=False
        )

    """
    def __init__(self, dataset, window=7, transforms=None, overlaps=True):
        self.dataset = dataset
        self.window = window

        if transforms is None:
            transforms = identity

        self.transforms = transforms
        self.overlaps = overlaps

        n = len(self)
        self.valid_size = int(n * 0.25)
        self.test_size = int(n * 0.25)
        self.train_size = n - self.valid_size - self.test_size

    def __getattr__(self, item):
        if hasattr(self.dataset, item):
            return getattr(self.dataset, item)

        return super(WindowedDataset, self).__getattr__(item)

    def __len__(self):
        if self.overlaps:
            return len(self.dataset) - self.window * 2

        return (len(self.dataset) - self.window) // self.window

    def __getitem__(self, item):
        if self.overlaps:
            return self.overlapping_get(item)

        return self.not_overlapping_get(item)

    def not_overlapping_get(self, item):
        # *----*----*
        #      *----*----*
        #           *----*----*
        s = item * self.window
        e = (item + 1) * self.window

        x = self.dataset[s:s + self.window, :]
        y = self.dataset[e:e + self.window, :]

        return self.transforms(x), self.transforms(y)

    def overlapping_get(self, item):
        """Returns two tensor representing n days before item and n days after item"""

        # *----*----*
        #  *----*----*
        #   *----*----*
        #    *----*----*
        item += self.window

        s = item - self.window
        e = item + self.window

        x = self.dataset[s:item, :]
        y = self.dataset[item:e, :]

        return self.transforms(x), self.transforms(y)
