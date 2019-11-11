from dataclasses import dataclass, field
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from olympus.metrics.metric import Metric
from olympus.utils.stat import StatStream


@dataclass
class Accuracy(Metric):
    loader: DataLoader = None
    accuracies: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    frequency_epoch: int = 1
    frequency_batch: int = 0
    name: str = 'validation'
    eval_time: StatStream = field(default_factory=lambda: StatStream(drop_first_obs=0))
    total_time: int = 0

    def state_dict(self):
        return dict(accuracies=self.accuracies, losses=self.losses)

    def load_state_dict(self, state_dict):
        self.accuracies = state_dict['accuracies']
        self.losses = state_dict['losses']

    def on_new_epoch(self, epoch, task, context):
        acc = 0
        loss_acc = 0

        start = datetime.utcnow()

        count = len(self.loader)
        for data, target in self.loader:
            accuracy, loss = task.accuracy(data, target)

            acc += accuracy.item()
            loss_acc += loss.item()

        end = datetime.utcnow()
        self.eval_time += (end - start).total_seconds()
        self.accuracies.append(acc / count)
        self.losses.append(loss_acc / count)

    def start(self, task=None):
        self.on_new_epoch(None, task, None)

    def finish(self, task=None):
        self.on_new_epoch(None, task, None)

    def value(self):
        if not self.accuracies:
            return {}

        return {
            f'{self.name}_accuracy': self.accuracies[-1],
            f'{self.name}_loss': self.losses[-1],
            f'{self.name}_time': self.eval_time.avg
        }


@dataclass
class OnlineTrainAccuracy(Metric):
    """Reuse precomputed loss and prediction to get accuracy
    because the model is updated in between each batch, this does not return the true accuracy on the training set,
    """
    accuracies: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    accumulator: int = 0
    loss: int = 0
    count: int = 0

    frequency_epoch: int = 1
    frequency_batch: int = 1

    def state_dict(self):
        return dict(
            accuracies=self.accuracies,
            losses=self.losses,
            accumulator=self.accumulator,
            loss=self.loss,
            count=self.count
        )

    def load_state_dict(self, state_dict):
        self.accuracies = state_dict['accuracies']
        self.losses = state_dict['losses']
        self.accumulator = state_dict['accumulator']
        self.loss = state_dict['loss']
        self.count = state_dict['count']

    def on_new_batch(self, step, task, input, context):
        _, targets = input
        predictions = context.get('predictions')

        # compute accuracy for the current batch
        if predictions is not None:
            _, predicted = torch.max(predictions, 1)

            target = input[1].to(device=task.device)

            loss = task.criterion(predictions, target).item()
            acc = (predicted == target).sum().item() / target.size(0)

            self.accumulator += acc
            self.loss += loss
            self.count += 1

    def on_new_epoch(self, epoch, task, context):
        if self.count > 0:
            # new epoch
            self.accuracies.append(self.accumulator / self.count)
            self.losses.append(self.loss / self.count)
            self.accumulator = 0
            self.loss = 0
            self.count = 0

    def finish(self, task=None):
        if self.count > 0:
            self.on_new_epoch(None, None, None)

    def value(self):
        if not self.accuracies:
            return {}

        return {
            'online_train_accuracy': self.accuracies[-1],
            'online_train_loss': self.losses[-1]
        }