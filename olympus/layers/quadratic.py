import torch
import torch.nn as nn


class Quadratic(nn.Module):
    """Compute ``X A X^T``.

    Attributes
    ----------
    A: Tensor(n x n)
        Defaults to the identity vector

    Examples
    --------

    >>> q = Quadratic((10, 20))
    >>> batch_size = 3
    >>> x = torch.randn((batch_size, 10, 20))
    >>> result = q(x)
    >>> result.shape
    torch.Size([3, 10, 10])
    """
    def __init__(self, input_shape, eps=1e-1):
        super(Quadratic, self).__init__()
        n, m = input_shape
        self.weight = nn.Parameter(torch.ones(m, dtype=torch.float32))
        self.A = nn.Parameter(torch.eye(m, dtype=torch.float32) + eps * torch.randn(m, m).float())

    def forward(self, x):
        """

        Parameters
        ----------
        x: N x S x T

        Returns
        -------
        A tensor of shape (N x S x S)
        """
        x = x * self.weight.unsqueeze(0)
        xt = x.transpose(2, 1)
        r1 = torch.matmul(x, self.A)
        return torch.matmul(r1, xt) / self.A.sum()
