import torch
import torch.nn as nn


class Differential(nn.Module):
    """Compute a 1d convolution with a default filter of [1, -1]. It is used to compute
    log returns in finance.

    Parameters
    ----------
    out_channel: int
        Number of output, if != input it means the network will create buckets of assets

    Notes
    -----
    Recent timesteps are last

    Examples
    --------

    >>> diff = Differential((10, 20), k=3, eps=0)
    >>> batch_size = 3
    >>> x = torch.randn((batch_size, 10, 20))
    >>> result = diff(x)
    >>> result.shape
    torch.Size([3, 10, 18])
    >>> torch.abs(result[0, 0, 0] - (x[0, 0, 2] - x[0, 0, 1])) < 1e-4
    tensor(True)
    """
    def __init__(self, input_shape, k=2, out_channel=None, eps=1e-1):
        super(Differential, self).__init__()
        n, _ = input_shape

        filter = torch.tensor([0 for _ in range(k)]).float()
        filter[-2] = -1
        filter[-1] = 1

        if out_channel is None:
            out_channel = n

        self.out_channel = out_channel
        in_channel = n
        kernel = torch.zeros(out_channel, in_channel, k).float()

        for i in range(n):
            kernel[i, i, :] = filter

        kernel = kernel + eps * torch.randn(out_channel, in_channel, k).float()
        self.kernel = nn.Parameter(kernel, requires_grad=True)

    def forward(self, x):
        """
        Parameters
        ----------
        x: N x S x T

        Returns
        -------
        A tensor of shape (N x S x (T - (k - 1)))
        """
        return nn.functional.conv1d(x, self.kernel)
