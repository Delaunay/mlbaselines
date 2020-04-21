from dataclasses import dataclass, field
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from olympus.observers.observer import Metric
from olympus.utils.stat import StatStream
from olympus.utils.cuda import Stream, stream


@dataclass
class Accuracy(Metric):
    loader: DataLoader = None
    accuracies: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    name: str = 'validation'
    eval_time: StatStream = field(default_factory=lambda: StatStream(drop_first_obs=0))
    total_time: int = 0
    metric_stream: Stream = field(default_factory=Stream)

    frequency_new_epoch: int = 1
    frequency_new_batch: int = 0

    def state_dict(self):
        return dict(accuracies=self.accuracies, losses=self.losses)

    def load_state_dict(self, state_dict):
        self.accuracies = state_dict['accuracies']
        self.losses = state_dict['losses']

    def compute_accuracy(self, task):
        start = datetime.utcnow()
        losses = []
        accs = []

        count = len(self.loader)

        with stream(self.metric_stream):
            with torch.no_grad():
                for data, target, *_ in self.loader:
                    accuracy, loss = task.accuracy(data, target)

                    accs.append(accuracy.detach())
                    losses.append(loss.detach())

                acc = sum([a.item() for a in accs])
                loss_acc = sum([l.item() for l in losses])

        end = datetime.utcnow()

        eval_time = (end - start).total_seconds()
        acc = (acc / count)
        loss = (loss_acc / count)

        return eval_time, acc, loss

    def on_new_epoch(self, task, epoch, context):
        # I would like to make this completely async
        # but I do not think I can do it easily
        # Good enough for now
        eval_time, acc, loss = self.compute_accuracy(task)

        self.eval_time += eval_time
        self.accuracies.append(acc)
        self.losses.append(loss)

    def on_start_train(self, task, step=None):
        self.on_new_epoch(task, step, None)

    def on_end_train(self, task, step=None):
        self.on_new_epoch(task, step, None)

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

    frequency_end_epoch: int = 1
    frequency_end_batch: int = 1

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

    def on_end_batch(self, task, step, input, context):
        _, targets, *_ = input
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

    def on_end_epoch(self, task, epoch, context):
        if self.count > 0:
            # new epoch
            self.accuracies.append(self.accumulator / self.count)
            self.losses.append(self.loss / self.count)
            self.accumulator = 0
            self.loss = 0
            self.count = 0

    def on_end_train(self, task, step=None):
        if self.count > 0:
            self.on_new_epoch(task, None, None)

    def value(self):
        if not self.accuracies:
            return {}

        return {
            'online_train_accuracy': self.accuracies[-1],
            'online_train_loss': self.losses[-1]
        }
