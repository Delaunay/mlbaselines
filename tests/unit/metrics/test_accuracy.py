import torch
import torch.nn.functional as f

from olympus.observers import ObserverList
from olympus.metrics import Accuracy
from olympus.utils import set_seeds


set_seeds(0)
BATCH_SIZE = 4

x = [torch.zeros((BATCH_SIZE, 100))]
y = torch.randint(0, 10, (BATCH_SIZE,))


class TaskMock:
    def __init__(self, callback=lambda e, i: None, epochs=12, steps=12):
        self.metrics = ObserverList()
        self.callback = callback
        self.epochs = epochs
        self.steps = steps
        self.metrics.task = self

    def fit(self):
        self.metrics.start_train()

        for e in range(0, self.epochs):
            self.metrics.new_epoch(e + 1)
            for i in range(0, self.steps):
                self.metrics.new_batch(i, input=x)
                self.callback(e, i)
                self.metrics.end_batch(i, input=x)
            self.metrics.end_epoch(e + 1)

        self.metrics.end_train()

    def accuracy(self, x, y):
        p = x[0].view(-1, 10, 10).sum(dim=2)
        scores = f.softmax(p, dim=1)
        _, predicted = torch.max(scores, 1)

        acc = (predicted == y).sum()
        loss = f.cross_entropy(scores, y)
        return acc.float(), loss


def test_accuracy():
    # Speed drop the first 5 observations
    loader = [(x, y) for _ in range(10)]
    acc = Accuracy(loader=loader, name='validation')
    task = TaskMock()
    task.metrics.append(acc)
    task.fit()

    assert acc.value()['validation_accuracy'] == 0.25
    assert acc.value()['validation_error_rate'] == 0.75
    assert acc.value()['validation_loss'] - 2.30258 < 1e-4


