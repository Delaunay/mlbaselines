import random
from dataclasses import dataclass

from olympus.metrics.metric import Metric
from olympus.baselines.classification import classification_baseline
from olympus.utils import fetch_device
from olympus.utils.storage import StateStorage
from olympus.utils.options import option


class Interrupt(Exception):
    pass


@dataclass
class InterruptingMetric(Metric):
    frequency_epoch: int = 1
    frequency_batch: int = 0

    def state_dict(self):
        return dict()

    def load_state_dict(self, state_dict):
        pass

    def on_new_batch(self, step, task, input, context):
        pass

    def on_new_epoch(self, epoch, task, context):
        if random.random() > 0.5:
            raise Interrupt()

    def finish(self, task=None):
        pass

    def value(self):
        return {}


def test_resume():
    device = fetch_device()
    state_storage = StateStorage(folder=option('state.storage', '/tmp'), time_buffer=30)

    params = {}
    task_no_interrupt = classification_baseline('logreg', 'glorot_uniform', 'sgd', 'none', 'mnist-test', 32, device)

    task_no_interrupt.init(**params)
    task_no_interrupt.fit(epochs=5)

    netrics1 = task_no_interrupt.metrics.value()

    task_resume = classification_baseline('logreg', 'glorot_uniform', 'sgd', 'none', 'mnist-test', 32, device)

    task_resume.init(**params)

    while True:
        task_resume.fit(epochs=5)

    netrics2 = task_no_interrupt.metrics.value()

