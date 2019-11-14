from dataclasses import dataclass, field
from datetime import datetime
from typing import List

import torch
from torch.utils.data import DataLoader

from olympus.metrics.metric import Metric
from olympus.utils.stat import StatStream


@dataclass
class Loss(Metric):
    loader: DataLoader = None
    losses: list = field(default_factory=list)
    frequency_epoch: int = 1
    frequency_batch: int = 0
    name: str = 'validation'
    eval_time: StatStream = field(default_factory=lambda: StatStream(drop_first_obs=0))
    total_time: int = 0

    def state_dict(self):
        return dict(losses=self.losses)

    def load_state_dict(self, state_dict):
        self.losses = state_dict['losses']

    def on_new_epoch(self, epoch, task, context):
        task.model.eval()

        with torch.no_grad():
            start = datetime.utcnow()
            count = len(self.loader)
            losses = []

            for batch in self.loader:
                losses.append(task.eval_loss(batch))

            # use item() to force the sync once all the job is sent to the GPU
            total = sum([loss.item() for loss in losses]) / count
            self.losses.append(total)
            end = datetime.utcnow()
            self.eval_time += (end - start).total_seconds()

        task.model.train()

    def start(self, task=None):
        self.on_new_epoch(None, task, None)

    def finish(self, task=None):
        self.on_new_epoch(None, task, None)

    def value(self):
        if not self.losses:
            return {}

        return {
            f'{self.name}_loss': self.losses[-1],
            f'{self.name}_time': self.eval_time.avg
        }


@dataclass
class OnlineLoss(Metric):
    losses: list = field(default_factory=list)
    loss_accumulator: List[float] = field(default_factory=list)

    frequency_epoch: int = 1
    frequency_batch: int = 1

    def state_dict(self):
        return dict(
            losses=self.losses,
        )

    def load_state_dict(self, state_dict):
        self.losses = state_dict['losses']

    def on_new_batch(self, step, task, input, context):
        loss = context.get('loss')

        # compute accuracy for the current batch
        if loss is not None:
            self.loss_accumulator.append(loss)

    def on_new_epoch(self, epoch, task, context):
        if self.loss_accumulator:
            self.losses.append(
                sum(l.item() for l in self.loss_accumulator) / len(self.loss_accumulator))

    def finish(self, task=None):
        self.on_new_epoch(None, None, None)

    def value(self):
        if not self.losses:
            return {}

        return {
            'online_train_loss': self.losses[-1]
        }
