from dataclasses import dataclass, field
from olympus.observers.observer import Metric


@dataclass
class NamedMetric(Metric):
    """Retrieve a value from the context and track its progress"""
    name: str = None
    metrics: list = field(default_factory=list)
    accumulator = 0
    count = 0

    def on_new_epoch(self, epoch, task, context):
        self.metrics.append(self.accumulator / self.count)
        self.accumulator = 0
        self.count = 0

    def on_new_batch(self, step, task, input, context):
        value = context.get(self.name)
        if value is not None:
            self.accumulator += value
            self.count += 1

    def finish(self, task):
        if self.count > 0:
            self.on_new_epoch(None, None, None)

    def value(self):
        return {
            self.name: self.metrics[-1]
        }


NamedObserver = NamedMetric

