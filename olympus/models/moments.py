import torch
import torch.nn as nn

from olympus.models import Module

from olympus.layers.quadratic import Quadratic
from olympus.layers.barycenter import Barycenter
from olympus.layers.differential import Differential


class Covariance(nn.Module):
    """Compute the covariance matrix

    Parameters
    ----------
    input_shape: (S, T)
        Shape of the expected tensor (batch size should not be in the shape)

    k: int
        Number of time steps used to compute the returns (with k=2, return=x_t - x_(t - 1))

    bias: bool
        If true use T to compute the covariance, else use T - 1

    Notes
    -----
    The initialization defaults the standard covariance formula.
    During training the weights change in unpredictable ways.

    Examples
    --------
    >>> cov = Covariance((10, 20), k=3, bias=True, eps=0)
    >>> batch_size = 3
    >>> x = torch.randn((batch_size, 10, 20))
    >>> result = cov(x)
    >>> result.shape
    torch.Size([3, 10, 10])
    >>> import numpy as np
    >>> ret = (x[0, :, 2:] - x[0, :, 1:-1]).numpy()
    >>> numpy_cov = np.cov(ret, bias=True)
    >>> torch.abs(result[0, :, :] - torch.from_numpy(numpy_cov)).sum() < 1e-5
    tensor(True)

    >>> cov = Covariance((10, 20), k=3, channels=100, bias=True)
    >>> batch_size = 3
    >>> x = torch.randn((batch_size, 10, 20))
    >>> result = cov(x)
    >>> result.shape
    torch.Size([3, 100, 100])
    """
    def __init__(self, input_shape, output_size=None, k=2, channels=None, bias=True, eps=1e-4):
        super(Covariance, self).__init__()

        n, m = input_shape

        self.diff = Differential((n, m), k=k, out_channel=channels, eps=eps)   # x_t - x_{t - 1}
        n = self.diff.out_channel

        self.quadratic = Quadratic((n, m - (k - 1)), eps=eps)  # r * r^T
        self.mean: Barycenter = Barycenter((n, m - (k - 1)))               # r * 1

        self.adapter = nn.Linear(n * n, n * n, bias=False)
        self.adapter.weight = nn.Parameter(torch.eye(n * n, dtype=torch.float), requires_grad=True)

        self.n = n
        self.size = n * n

    def forward(self, x):
        returns = self.diff(x)

        squared = self.quadratic(returns)
        mean = self.mean(returns)

        cov = squared - torch.matmul(mean, mean.transpose(2, 1))

        # go through a single fully connected layer
        flat_cov = cov.view(-1, self.size)
        cov = self.adapter(flat_cov).view(-1, self.n, self.n)
        return cov

    def smooth(self):
        self.mean.smooth()

    def init(self, **kwargs):
        pass

    @staticmethod
    def get_space():
        return {
            'k': 'uniform(2, 60, discrete=True)',
        }


class Mean(nn.Module):
    def __init__(self, input_shape, k=2, channels=None, eps=1e-4, **kwargs):
        super(Mean, self).__init__()
        n, m = input_shape
        self.diff = Differential((n, m), k=k, out_channel=channels, eps=eps)  # x_t - x_{t - 1}
        self.mean = Barycenter((n, m - (k - 1)))
        self.adapter = nn.Linear(n, n, bias=False)
        self.adapter.weight.data = torch.eye(n, dtype=torch.float32)
        self.n = n

    def forward(self, x):
        returns = self.diff(x)
        x = self.mean(returns)
        x = x.view(-1, self.n)
        x = self.adapter(x)
        return x.view(-1, self.n, 1)

    @staticmethod
    def get_space():
        return {
            'k': 'uniform(2, 60, discrete=True)',
        }


class MinVariance(nn.Module):
    def __init__(self, num, estimator):
        super(MinVariance, self).__init__()
        self.cov_estimator = estimator
        self.n = num

    def forward(self, input):
        """Returns the target weight in %"""
        batch_size, _, _ = input.shape
        cov = self.cov_estimator(input)

        A = torch.zeros((batch_size, self.n + 1, self.n + 1))
        A[:, self.n, 0:self.n] = 1
        A[:, 0:self.n, self.n] = - 1
        A[:, 0:self.n, 0:self.n] = cov

        B = torch.zeros((batch_size, self.n + 1, 1))
        B[:, self.n] = 1

        x, _ = torch.solve(B, A)
        return x[:, 0:self.n]


class MinVarianceReturn(nn.Module):
    def __init__(self, num, cov_estimator, return_estimator, target=None):
        super(MinVarianceReturn, self).__init__()
        self.cov_estimator = cov_estimator
        self.return_estimator = return_estimator
        self.n = num

        if target is None:
            target = 0.12 / 365.0
        else:
            target = float(target)

        # this will be optimized as well as part of maximizing the sharp ratio
        self.returns = nn.Parameter(torch.scalar_tensor(target, requires_grad=True))

    def share_weights(self):
        """Make all sample weight be shared across estimators"""
        if isinstance(self.cov_estimator, Covariance):
            master_weight = self.cov_estimator.quadratic.weight
            self.cov_estimator.mean.weight = master_weight

            if isinstance(self.return_estimator, Mean):
                self.return_estimator.mean.weight = master_weight

    def forward(self, input):
        """Returns the target weight in %"""
        batch_size, _, _ = input.shape

        cov = self.cov_estimator(input)
        ret = self.return_estimator(input).squeeze(2)

        A = torch.zeros((batch_size, self.n + 2, self.n + 2))
        A[:, self.n, 0:self.n] = + 1
        A[:, 0:self.n, self.n] = - 1

        A[:, self.n + 1, 0:self.n] = ret
        A[:, 0:self.n, self.n + 1] = - ret

        A[:, 0:self.n, 0:self.n] = cov

        B = torch.zeros((batch_size, self.n + 2, 1))
        B[:, self.n] = 1
        B[:, self.n + 1] = self.returns

        x, _ = torch.solve(B, A)
        return x[:, 0:self.n]


class MinVarianceReturnMomentEstimator(Module):
    """The rational behind this estimator is to use standard estimators such as Covariance and Mean but enable
    the network to tweak each steps of the computation to make them forward looking.
    We expect most weights to not change much from the default
    """
    def __init__(self, input_size, output_size, lags):
        super(MinVarianceReturnMomentEstimator, self).__init__()
        n, m = input_size

        self.cov = Covariance(input_size, k=lags)
        self.mean = Mean(input_size, k=lags)
        self.model = MinVarianceReturn(n, self.cov, self.mean)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def get_space():
        return {
            'lags': 'uniform(2, 64, discrete=True)',
        }


builders = {
    'MinVarianceReturnMomentEstimator': MinVarianceReturnMomentEstimator
}
