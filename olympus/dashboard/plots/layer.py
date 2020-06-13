import math
import torch.nn as nn
import matplotlib.pyplot as plt

plt.style.use('dark_background')


def vizualize_param(p, ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    def heatmap(data):
        im = ax.imshow(data.numpy())

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('', rotation=-90, va="bottom")

    def histogram(data):
        ax.bar(x=range(len(data)), height=data)

    p = p.squeeze()

    if len(p.shape) == 3:
        for i in range(p.shape[0]):
            heatmap(p[i, :, :].detach())

    elif len(p.shape) == 4:
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                heatmap(p[i, j, :, :].detach())

    elif len(p.shape) == 2:
        heatmap(p[:, :].detach())

    elif len(p.shape) == 1:
        histogram(p[:].detach())

    return fig


def vizualize_weights(module: nn.Module):
    params = list(module.parameters())

    n = 0
    for p in params:
        shape = p.shape
        if len(shape) == 3:
            n += shape[0]
        elif len(shape) == 4:
            n += shape[0] * shape[1]
        elif len(shape) > 4:
            print('skip high dimension weight ', shape)
        else:
            n += 1

    if n < 4:
        row = n
        col = 1
    else:
        n = int(math.sqrt(n)) + 1
        row = n
        col = n

    fig = plt.figure()
    gs = fig.add_gridspec(row, col)
    k = 0

    def get_col(k):
        if col > 1:
            return k % row
        return 0

    def get_row(k):
        if row > 1:
            return k // col
        return 0

    for p in params:
        ax = fig.add_subplot(gs[get_row(k), get_col(k)])
        vizualize_param(p, ax)
        k += 1

    return fig


if __name__ == '__main__':
    fig = vizualize_weights(nn.Conv2d(3, 10, 3))
    plt.show()
