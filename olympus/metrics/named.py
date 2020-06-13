import torch

from dataclasses import dataclass, field
from olympus.observers.observer import Metric


@dataclass
class NamedMetric(Metric):
    """Retrieve a value from the context and track its progress"""
    name: str = None
    metrics: list = field(default_factory=list)
    accumulator = 0
    count = 0

    frequency_end_batch: int = 1
    frequency_end_epoch: int = 1

    def start(self, task=None):
        pass

    def on_end_epoch(self, task, epoch, context):
        self.metrics.append(self.accumulator / self.count)
        self.accumulator = 0
        self.count = 0

    def on_end_batch(self, task, step, input, context):
        value = context.get(self.name)

        if isinstance(value, torch.Tensor):
            value = value.item()

        if value is not None:
            self.accumulator += value
            self.count += 1

    def on_end_train(self, task, step=None):
        if self.count > 0:
            self.on_end_epoch(task, None, None)

    def value(self):
        if not self.metrics:
            return {}

        return {
            self.name: self.metrics[-1]
        }

    def state_dict(self):
        return {
            self.name: self.metrics
        }

    def load_state_dict(self, state_dict):
        self.metrics = state_dict[self.name]
        self.accumulator = 0
        self.count = 0


NamedObserver = NamedMetric

