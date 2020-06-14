import torch
import torch.nn as nn
from torch.optim import SGD
from sspace.space import hyperparameter, uniform

from random import Random

from olympus.observers import metric_logger

import time


@hyperparameter(lr=uniform(0, 1), b=uniform(0, 1), c=uniform(0, 1))
def my_trial(epoch, lr, a, b, c, **kwargs):
    import time
    time.sleep(epoch / 10)
    return lr * a - b * c


def _data(seed, features=32):
    r = Random(seed)
    weights = torch.as_tensor([[r.uniform(0, 1) for i in range(features)] for _ in range(2)]).t()

    with torch.no_grad():
        x = torch.randn((1024, features))
        y = x.matmul(weights)
        return x.cuda(), y.cuda()


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


def tiny_task(batch_size, epochs, lr, seed, experiment_name, client=None, uid=None):
    x, y = _data(seed)
    n, features = x.shape

    model = LinearRegression(features, 2).cuda()
    optimizer = SGD(params=model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    logger = metric_logger(
        uri='mongo://127.0.0.1:27017',
        database='olympus',
        experiment=experiment_name, client=client)
    logger.client.uid = uid

    val_loss = None

    for e in range(epochs):
        losses = []
        for i in range(n // batch_size):
            input = x[i * batch_size:(i + 1) * batch_size, :]
            target = y[i * batch_size:(i + 1) * batch_size, :]

            prediction = model(input)
            loss = criterion(target, prediction)
            losses.append(loss.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        time.sleep(1)
        loss_epoch = sum([l.cpu().item() for l in losses]) / (n // batch_size)

        x_val, y_val = _data(seed)
        val_loss = criterion(y_val, model(x_val)).cpu().item()

        logger.log(
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            epoch=e,
            loss_epoch=loss_epoch,
            val_loss=val_loss)

    return val_loss
