from dataclasses import dataclass, field
from datetime import datetime

import torch
from torch.utils.data import DataLoader

import numpy as np

from olympus.observers.observer import Metric
from olympus.utils import error
from olympus.utils.stat import StatStream
from olympus.utils.cuda import Stream, stream


class NotFittedError(Exception):
    pass


def detach(f):
    if isinstance(f, torch.Tensor):
        return f.detach()
    return f


def item(f):
    if isinstance(f, torch.Tensor):
        return f.item()
    return f


@dataclass
class MeanIoU(Metric):
    loader: DataLoader = None
    mean_accs: list = field(default_factory=list)
    pixel_accs: list = field(default_factory=list)
    meanIoUs: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    name: str = 'validation'
    eval_time: StatStream = field(default_factory=lambda: StatStream(drop_first_obs=0))
    total_time: int = 0
    metric_stream: Stream = field(default_factory=Stream)

    frequency_new_epoch: int = 1
    frequency_new_batch: int = 0

    def state_dict(self):
        return dict(meanIoUs=self.meanIoUs, pixel_accs=self.pixel_accs, mean_accs=self.mean_accs, losses=self.losses)

    def load_state_dict(self, state_dict):
        self.mean_accs = state_dict['mean_accs']
        self.pixel_accs = state_dict['pixel_accs']
        self.meanIoUs = state_dict['meanIoUs']
        self.losses = state_dict['losses']

    def compute_meanIoU(self, task):
        start = datetime.utcnow()
        with stream(self.metric_stream):
            losses = []
            confusion_matrix = np.zeros((task.nclasses,task.nclasses))
            with torch.no_grad():
                for data, target in self.loader:
                    conf_mtx, loss = task.confusion_matrix(data, target)
                    confusion_matrix += conf_mtx
                    losses.append(detach(loss))

            # Per-Class Statistics
            class_ntp = np.diagonal(confusion_matrix)
            class_freq = np.sum(confusion_matrix, axis=1)
            prediction_freq = np.sum(confusion_matrix, axis=0)

            # Per-Class Metrics
            class_acc = class_ntp / class_freq
            mean_acc = np.mean(class_acc)
            classIoU = class_ntp / (class_freq + prediction_freq - class_ntp)
            meanIoU = np.mean(classIoU)

            # Global Metrics
            pixel_acc = np.sum(class_ntp) / np.sum(class_freq)

            # Mean Loss
            loss = item(np.mean(losses))
        end = datetime.utcnow()
        eval_time = (end - start).total_seconds()
        return eval_time, mean_acc, pixel_acc, meanIoU, loss

    def get_meanIoU(self, task, epoch, context):
        eval_time, mean_acc, pixel_acc, meanIoU, loss = self.compute_meanIoU(task)

        self.eval_time += eval_time
        self.mean_accs.append(mean_acc)
        self.pixel_accs.append(pixel_acc)
        self.meanIoUs.append(meanIoU)
        self.losses.append(loss)

    def on_end_epoch(self, task, epoch, context):
        pass

    def on_end_train(self, task, step=None):
        self.get_meanIoU(task, step, None)

    def on_start_train(self, task, step=None):
        pass

    def value(self):
        if not self.meanIoUs or not self.pixel_accs or not self.mean_accs:
            return {}

        return {
            f'{self.name}_mean_acc': self.mean_accs[-1],
            f'{self.name}_pixel_acc': self.pixel_accs[-1],
            f'{self.name}_meanIoU': self.meanIoUs[-1],
            f'{self.name}_mean_jaccard_distance': 1 - self.meanIoUs[-1],
            f'{self.name}_loss': self.losses[-1],
            f'{self.name}_time': self.eval_time.avg
        }
