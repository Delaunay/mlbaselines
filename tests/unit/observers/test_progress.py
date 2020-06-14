import numpy as np
import time
import pytest
import os
from olympus.observers import ObserverList, Speed, ProgressView, ElapsedRealTime
from olympus.observers import SampleCount

BATCH_SIZE = 4
x = [np.zeros((BATCH_SIZE, 3, 224, 224))]


#
# os.environ['OLYMPUS_PROGRESS_FREQUENCY_EPOCH'] = '0'
# os.environ['OLYMPUS_PROGRESS_FREQUENCY_BATCH'] = '0'


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


def stop_after(stop_epoch, stop_batch, sleep_time=1):
    def callback(e, i):
        epoch_stop = stop_epoch is not None and e == stop_epoch
        batch_stop = stop_batch is not None and i == stop_batch
        time.sleep(sleep_time)

        if stop_epoch is None and batch_stop:
            raise StopIteration

        if stop_batch is None and epoch_stop:
            raise StopIteration

        if batch_stop and epoch_stop:
            raise StopIteration

        return
    return callback


def test_speed_1():
    # Speed drop the first 5 observations
    speed = Speed()
    task = TaskMock(callback=stop_after(None, 6, sleep_time=1))
    task.metrics.append(speed)

    with pytest.raises(StopIteration):
        task.fit()

    assert speed.step_time.avg - 1 <= 0.1
    assert speed.step_time.sd == 0
    assert speed.step_time.avg == speed.step_time.max
    assert speed.step_time.avg == speed.step_time.min
    assert speed.step_time.count == 1


def test_speed_5():
    # Speed drop the first 5 observations
    speed = Speed()
    progress = ProgressView(speed)

    task = TaskMock(callback=stop_after(None, 10, sleep_time=1))
    task.metrics.append(speed)
    task.metrics.append(progress)

    with pytest.raises(StopIteration):
        task.fit()

    assert speed.step_time.avg - 1 <= 0.1
    assert speed.step_time.sd <= 0.1
    assert speed.step_time.count == 5
    assert speed.value()['batch_speed'] - 4 <= 0.1


def test_sample_count():
    # Speed drop the first 5 observations
    count = SampleCount()

    task = TaskMock(epochs=12, steps=12)
    task.metrics.append(count)
    task.fit()

    assert count.value()['sample_count'] == BATCH_SIZE * 12 * 12


def test_elapsed_real_time():
    # Speed drop the first 5 observations
    timer = ElapsedRealTime()

    task = TaskMock(
        epochs=12,
        steps=12,
        callback=stop_after(None, None, sleep_time=0.01))
    task.metrics.append(timer)
    task.fit()

    assert timer.value()['elapsed_time'] - 12 * 12 * 0.01 < 0.1


def named_print(name):
    def new_print(*args, end='\n'):
        args = [a.replace('\r', '') for a in args]
        print(name, *args, end='\n')
    return new_print


def show_progress():
    epochs = 4
    steps = 12

    # Speed drop the first 5 observations
    speed = Speed()
    progress_default = ProgressView(speed)
    progress_epoch_guess = ProgressView(speed, max_epochs=epochs)
    progress_epoch = ProgressView(speed, max_epochs=epochs, max_steps=steps)
    progress_steps = ProgressView(speed, max_steps=epochs * steps)

    progress_default.print_fun     = named_print('default')
    progress_epoch_guess.print_fun = named_print('  guess')
    progress_epoch.print_fun       = named_print('  epoch')
    progress_steps.print_fun       = named_print('   step')

    task = TaskMock(
        callback=stop_after(None, None, sleep_time=1),
        epochs=epochs,
        steps=steps
    )

    task.metrics.append(speed)
    task.metrics.append(ElapsedRealTime())
    task.metrics.append(progress_default)
    task.metrics.append(progress_epoch_guess)
    task.metrics.append(progress_epoch)
    task.metrics.append(progress_steps)
    task.fit()


if __name__ == '__main__':
    # progress.show.metrics
    os.environ['OLYMPUS_PROGRESS_SHOW_METRICS'] = 'epoch'

    show_progress()
