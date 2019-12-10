import random
import os
import sys
from typing import Callable
from dataclasses import dataclass, field

from olympus.utils import fetch_device, set_verbose_level
set_verbose_level(60)

from olympus.observers.observer import Metric
from olympus.baselines.classification import classification_baseline
from olympus.utils.storage import StateStorage, NoStorage
from olympus.resuming import BadResume

import pytest
from itertools import zip_longest


class Interrupt(Exception):
    pass


interruption_counter = 0
interruption_counter_batch = 0
max_interrupts = 30
max_interrupts_batch = 10
model_parameters1 = None
params = {
    'optimizer': {
        'lr': 0.011113680070144951,
        'momentum': 0.04081791544572477,
        'weight_decay': 6.2091793568732874e-06
    },
}
keys = [
    'online_train_accuracy',
    'validation_loss',
    'validation_accuracy',
    'online_train_loss',
    'online_train_accuracy',
    'epoch',
    'sample_count'
]


class FakeTrial:
    id = 0
    hash_params = 'checkpoint.state'


def random_interrupt(epoch):
    global interruption_counter
    a = random.random()

    should = a > 0.5 and interruption_counter < max_interrupts
    if should:
        interruption_counter += 1

    return should


def random_interrupt_batch(epoch):
    global interruption_counter_batch
    a = random.random()

    should = a > 0.5 and interruption_counter_batch < max_interrupts_batch
    if should:
        interruption_counter_batch += 1

    return should


def remove(filename):
    try:
        os.remove(filename)
    except:
        pass


def compare(d1, d2, depth=0):
    import torch
    from collections import defaultdict

    if type(d1) != type(d2):
        print(f'({d1} {d2})', end=' ')
        return False

    elif isinstance(d1, (dict, defaultdict)):
        keys = set(d1.keys())
        keys.update(set(d2.keys()))
        keys = list(keys)

        for k in keys:
            v1 = d1.get(k)
            v2 = d2.get(k)

            end = ''
            if isinstance(v1, dict):
                end = '\n'

            print(f'{" " * depth} {k:<30}: ', end=end)
            compare(v1, v2, depth + 1)
            print()

    elif isinstance(d1, list):
        for v1, v2 in zip_longest(d1, d2):
            compare(v1, v2, depth + 1)

    elif isinstance(d1, (float, int)):
        print(f'{float(d1):15f} | {float(d2):15f} | {float(d1 - d2):15f}', end=' ')

    elif isinstance(d1, torch.Tensor):
        print((d1 - d2).std().item(), end='  ')
    else:
        print(type(d1), d1, d2)

    return True


@dataclass
class InterruptingMetric(Metric):
    frequency_epoch: int = 1
    frequency_batch: int = 0
    interrupt_schedule_epoch: Callable = random_interrupt
    interrupt_schedule_batch: Callable = random_interrupt_batch
    # Interrupt after Checkpointing
    priority: int = -100
    epoch: int = field(default=0)

    def state_dict(self):
        return dict(epoch=self.epoch + 1)

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']

    def on_new_batch(self, step, task, input, context):
        print(self.epoch)
        if self.epoch > 0 and self.interrupt_schedule_batch(step):
            print('Interrupting Batch')
            raise Interrupt()

    def on_new_epoch(self, epoch, task, context):
        self.epoch = epoch
        if self.interrupt_schedule_epoch(epoch):
            print(f'Interrupting Epoch {self.epoch} {self.frequency_batch} {self.frequency_epoch}')
            raise Interrupt()

    def start(self, task=None):
        self.epoch = 0

    def finish(self, task=None):
        pass

    def value(self):
        return {}


def make_base_task(device, storage):
    return classification_baseline(
        'logreg', 'glorot_uniform',
        'sgd', 'none', 'test-mnist', 32,
        device, storage=storage)


def run_no_interrupts(epoch, params, device):
    global model_parameters1

    task_no_interrupt = make_base_task(device, NoStorage())

    task_no_interrupt.init(**params)
    chk = task_no_interrupt.metrics.get('CheckPointer')
    print('no_interrupts_id: ', chk.trial_id)

    model_parameters1 = list(task_no_interrupt.model.parameters())
    task_no_interrupt.fit(epochs=epoch)
    return task_no_interrupt.metrics.value()


def run_with_interrupts(epoch, batch_freq, state_folder, params, device):
    print('=' * 80)
    state_storage = StateStorage(folder=state_folder, time_buffer=0)

    def make_task():
        task_resume = make_base_task(device, storage=state_storage)
        task_resume.metrics.append(InterruptingMetric().every(epoch=1, batch=batch_freq))
        return task_resume

    def get_trial_id(task):
        return task.metrics.get('CheckPointer').trial_id

    def delete_state():
        task_resume = make_task()
        task_resume.init(**params)

        state_filename = f'{state_folder}/{get_trial_id(task_resume)}.state'
        remove(state_filename)

    delete_state()

    task_resume = make_task()
    task_resume.init(**params)
    assert not task_resume.resumed()

    trial_id = get_trial_id(task_resume)

    running = True
    while running:
        try:
            task_resume.fit(epochs=epoch)
            running = False

        except Interrupt:
            # Bad Resume, should not call load_state directly
            with pytest.raises(BadResume):
                state = state_storage.load(trial_id)
                assert state is not None
                task_resume.load_state_dict(state)

            # Init should resume automatically
            task_resume = make_task()
            task_resume.init(**params)

            chk = task_resume.metrics.get('CheckPointer')
            assert trial_id == chk.trial_id
            assert task_resume.resumed

    return task_resume.metrics.value()


def main_resume(epoch, batch_freq=0):
    global interruption_counter
    interruption_counter = 0

    state_folder = '/tmp/olympus/tests'

    device = fetch_device()

    stdout = sys.stdout
    # sys.stdout = open(os.devnull, "w")

    metrics1 = run_no_interrupts(epoch, params, device)
    metrics2 = run_with_interrupts(epoch, batch_freq, state_folder, params, device)

    # sys.stdout.close()
    sys.stdout = stdout

    print(f'epoch = {epoch}')
    print(f'interrupted = {interruption_counter}')
    print(f'interrupted = {interruption_counter_batch}')
    print(f'{"key":>30} | {"NoInterrupt":>12} | {"Interrupted":>12}')
    for k, v in metrics1.items():
        print(f'{k:>30} | {v:12.4f} | {metrics2.get(k, float("NaN")):12.4f}')

    for k in keys:
        diff = abs(metrics1[k] - metrics2[k])
        print(f'{k} => {diff}')
        assert diff < 1e-4, f'diff for {k} should be lower but it is {diff}'


def create_trained_trial(epochs=5):
    """Create a Task that was trained from scratch without interruption"""
    device = fetch_device()
    task = make_base_task(device, NoStorage())
    task.init(**params)
    task.fit(epochs=epochs)
    return task


def create_resumed_trained_trial(epochs=5):
    """Create a Task was trained stopped and resumed"""
    device = fetch_device()

    class FakeTrial2:
        id = 0
        hash_params = 'serialization_test'

    old_task = create_trained_trial(epochs)

    # Saves Task
    state_storage = StateStorage(folder='/tmp/olympus/tests', time_buffer=0)
    saved = state_storage.save('serialization_test', old_task.state_dict())
    assert saved

    new_task = make_base_task(device, state_storage)
    # Automatic Resume
    new_task.init(trial=FakeTrial2(), **params)
    assert new_task.resumed()
    return new_task


def test_model_serialization(epochs=5):
    """Check that models evaluate the same way after resume"""
    remove('/tmp/olympus/tests/serialization_test.state')
    original_trial = create_trained_trial(epochs)
    resumed_trial = create_resumed_trained_trial(epochs)

    print('-' * 80)
    compare(original_trial.state_dict(), resumed_trial.state_dict())
    print('-' * 80)

    original_acc = original_trial.metrics.get('validation')
    resumed_acc = resumed_trial.metrics.get('validation')

    # Metrics are the same
    original_m = original_acc.value()
    resumed_m = resumed_acc.value()
    for k in ['validation_accuracy', 'validation_loss']:
        assert abs(original_m[k] - resumed_m[k]) < 1e-4

    # Eval to the same result
    _, acc1, loss1 = original_acc.compute_accuracy(original_trial)
    _, acc2, loss2 = resumed_acc.compute_accuracy(resumed_trial)

    assert abs(acc1 - acc2) <= 1e-4
    assert abs(loss1 - loss2) <= 1e-4


def test_model_resume_train(epochs=5):
    """Check that models with and without resume train the same way"""
    remove('/tmp/olympus/tests/serialization_test.state')
    original_trial = create_trained_trial(epochs)
    resumed_trial = create_resumed_trained_trial(epochs)

    print(' {:<30}: {:>15} | {:>15} | {:>15}'.format('key', 'original', 'resumed', 'diff'))
    print('-' * 80)
    compare(
        original_trial.metrics.value(),
        resumed_trial.metrics.value()
    )

    print('---\nCompare Model Weights')
    compare(
        list(original_trial.model.parameters()),
        list(resumed_trial.model.parameters())
    )

    print('\nTRAIN')
    # Execute a few more steps
    stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

    original_trial.fit(epochs=epochs + 1)
    resumed_trial.fit(epochs=epochs + 1)

    sys.stdout.close()
    sys.stdout = stdout

    print(' {:<30}: {:>15} | {:>15} | {:>15}'.format('key', 'original', 'resumed', 'diff'))
    print('-' * 80)
    compare(
        original_trial.metrics.value(),
        resumed_trial.metrics.value()
    )

    print('Compare Model Weights')
    compare(
        list(original_trial.model.parameters()),
        list(resumed_trial.model.parameters())
    )
    print()


def task_deterministic(epoch=5):
    device = fetch_device()

    state_folder = '/tmp/olympus/tests'
    file_name = f'{state_folder}/94d2cd8af8fd709a4e7abcbde32fd9782b7e150fc9d3a3d6a10d98d07f247d56.state'

    metrics1 = run_no_interrupts(epoch, params, device)
    remove(file_name)

    metrics2 = run_no_interrupts(epoch, params, device)
    remove(file_name)

    for k in keys:
        diff = abs(metrics1[k] - metrics2[k])
        print(f'{k} => {diff}')
        assert diff < 1e-4


def test_task_deterministic():
    for i in range(1, 10):
        task_deterministic(i)


def test_task_resume():
    main_resume(epoch=1)
    main_resume(epoch=10)


if __name__ == '__main__':
    os.environ['OLYMPUS_DATA_PATH'] = '/tmp'

    test_model_serialization()
    test_model_resume_train(2)
    main_resume(20, 1)
    main_resume(5)
