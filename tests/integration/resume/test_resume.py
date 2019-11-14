import random
import os
import sys
from typing import Callable
from dataclasses import dataclass

from olympus.metrics.metric import Metric
from olympus.baselines.classification import classification_baseline
from olympus.utils import fetch_device, BadResume
from olympus.utils.storage import StateStorage, NoStorage

import pytest


class Interrupt(Exception):
    pass


interruption_counter = 0
max_interrupts = 30
model_parameters1 = None


def random_interrupt(epoch):
    global interruption_counter
    a = random.random()

    should = a > 0.5 and interruption_counter < max_interrupts
    if should:
        interruption_counter += 1

    return should


@dataclass
class InterruptingMetric(Metric):
    frequency_epoch: int = 1
    frequency_batch: int = 0
    interrupt_schedule: Callable = random_interrupt

    def state_dict(self):
        return dict()

    def load_state_dict(self, state_dict):
        pass

    def on_new_epoch(self, epoch, task, context):
        if self.interrupt_schedule(epoch):
            raise Interrupt()

    def finish(self, task=None):
        pass

    def value(self):
        return {}


def run_no_interrupts(epoch, params, device):
    global model_parameters1

    task_no_interrupt = classification_baseline(
        'logreg', 'glorot_uniform', 'sgd', 'none', 'test-mnist', 32, device, storage=NoStorage())

    task_no_interrupt.init(**params)

    model_parameters1 = list(task_no_interrupt.model.parameters())

    task_no_interrupt.fit(epochs=epoch)
    return task_no_interrupt.metrics.value()


def run_with_interrupts(epoch, state_folder, params, device):
    try:
        os.remove(f'{state_folder}/checkpoint.state')
    except:
        pass

    state_storage = StateStorage(folder=state_folder, time_buffer=0)

    def make_task():
        task_resume = classification_baseline(
            'logreg', 'glorot_uniform', 'sgd', 'none', 'test-mnist', 32, device, storage=state_storage)

        task_resume.metrics.append(InterruptingMetric())
        return task_resume

    task_resume = make_task()
    task_resume.init(**params)

    running = True
    while running:
        try:
            task_resume.fit(epochs=epoch)
            running = False

        except Interrupt:
            with pytest.raises(BadResume):
                task_resume.resume()

            # proper way to resume
            task_resume = make_task()
            task_resume.init(**params)

    return task_resume.metrics.value()


def main_resume(epoch):
    global interruption_counter
    interruption_counter = 0

    state_folder = '/tmp'

    device = fetch_device()

    stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

    params = {
        'optimizer': {
            'lr': 0.011113680070144951,
            'momentum': 0.04081791544572477,
            'weight_decay': 6.2091793568732874e-06
        },
    }

    metrics1 = run_no_interrupts(epoch, params, device)
    metrics2 = run_with_interrupts(epoch, state_folder, params, device)

    sys.stdout.close()
    sys.stdout = stdout

    print(f'epoch = {epoch}')
    print(f'interrupted = {interruption_counter}')
    print(f'{"key":>30} | {"NoInterrupt":>12} | {"Interrupted":>12}')
    for k, v in metrics1.items():
        print(f'{k:>30} | {v:12.4f} | {metrics2.get(k):12.4f}')

    keys = [
        'online_train_accuracy',
        'validation_loss',
        'validation_accuracy',
        'online_train_loss',
        'online_train_accuracy',
        'epoch',
        'sample_count'
    ]

    for k in keys:
        assert metrics1[k] == metrics2[k]


def test_task_resume():
    main_resume(epoch=1)
    main_resume(epoch=10)


if __name__ == '__main__':
    os.environ['OLYMPUS_DATA_PATH'] = '/tmp'
    main_resume(5)
