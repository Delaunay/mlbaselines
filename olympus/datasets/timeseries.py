import hashlib
import os

import torch
from olympus.datasets.dataset import AllDataset


class StockMarketDataset(AllDataset):
    """

    Examples
    --------

    .. code-block:: python

        dataset = StockMarketDataset(['AAPL'], '2000-01-01', '2019-05-10')

    """
    @staticmethod
    def cache_folder(data_path):
        if data_path is not None:
            os.makedirs(data_path, exist_ok=True)
            return data_path

        dirname = os.path.dirname(__file__)
        return os.path.join(dirname, '..', 'cache')

    @staticmethod
    def cache_key(*args):
        m = hashlib.sha256()

        for arg in args:
            if isinstance(arg, list):
                arg = StockMarketDataset.cache_key(*arg)
            m.update(arg.encode('utf-8'))

        return m.hexdigest()[:16]

    @staticmethod
    def fetch_data(tickers, start, end, source, data_path=None):
        import pandas as pd
        from pandas_datareader import data

        folder = StockMarketDataset.cache_folder(data_path)

        key = StockMarketDataset.cache_key(tickers, start, end, source)
        cache_file = os.path.join(folder, key)

        if os.path.exists(cache_file):
            aapl = pd.read_csv(cache_file, index_col='Date')
        else:
            aapl = data.DataReader(
                tickers,
                start=start,
                end=end,
                data_source=source)['Adj Close']

            aapl.to_csv(cache_file)

        cleaned = aapl.dropna()
        x = torch.from_numpy(cleaned.values)
        x = x.float().log()

        return x, cleaned

    def __init__(self, tickers, start_date, end_date, source='yahoo', data_path=None):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.source = source
        self.data, self.raw = self.fetch_data(tickers, start_date, end_date, source, data_path=data_path)

        n = self.data.shape[0]
        valid_size = int(n * 0.25)
        test_size  = int(n * 0.25)
        train_size = n - valid_size - test_size

        assert n == valid_size + test_size + train_size

        super(StockMarketDataset, self).__init__(self, data_path=None, input_shape=None, target_shape=None,
                                                 train_size=train_size, valid_size=valid_size, test_size=test_size)

    def __len__(self):
        """Returns the number of days inside our time series"""
        return self.data.shape[0]

    def __getitem__(self, item):
        """Returns an array of observation for a given day"""
        if isinstance(item, slice) or (isinstance(item, tuple) and isinstance(item[0], slice)):
            return self.data[item]

        return self.data[item, :]


builders = {
    'stockmarket': StockMarketDataset,
}

