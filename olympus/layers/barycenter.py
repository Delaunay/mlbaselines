import torch
import torch.nn as nn


class Barycenter(nn.Module):
    """Compute the barycenter of a sample or weighted average

    Attributes
    ----------
    weight: Tensor(T x 1)
        Defaults to 1 / T (equivalent to the mean)

    Examples
    --------

    >>> barycenter = Barycenter((10, 20))
    >>> batch_size = 3
    >>> x = torch.randn((batch_size, 10, 20))
    >>> result = barycenter(x)
    >>> result.shape
    torch.Size([3, 10, 1])
    >>> torch.abs(result - x.mean(dim=2).unsqueeze(2)).sum() < 1e-5
    tensor(True)

    """
    def __init__(self, input_shape):
        super(Barycenter, self).__init__()
        self.weight = nn.Parameter(torch.ones(input_shape[1], dtype=torch.float32))
        self.n = input_shape[1]

    def forward(self, x):
        return torch.matmul(x, self.weight.unsqueeze(1) / self.n)

    def smooth(self, steps=4):
        n = self.weight.shape[0]
        new_weight = torch.zeros_like(self.weight)

        for i in range(n):
            s = max(i - steps // 2, 0)
            e = min(i + steps // 2, n)
            new_weight[i] = self.weight[s:e].median()

        self.weight = nn.Parameter(new_weight / new_weight.sum())
