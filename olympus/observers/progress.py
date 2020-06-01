from dataclasses import dataclass, field

from olympus.observers.observer import Observer
from olympus.utils import show_dict, TimeThrottler
from olympus.utils.stat import StatStream
from olympus.utils.options import option

from datetime import datetime, timedelta
import warnings


def get_time_delta(start):
    return (datetime.utcnow() - start).total_seconds()


@dataclass
class Speed(Observer):
    batch_size: int = 0

    frequency_new_epoch: int = 1
    frequency_end_epoch: int = 1
    frequency_new_batch: int = 1
    frequency_end_batch: int = 1

    step_time: StatStream = field(default_factory=lambda: StatStream(drop_first_obs=5))
    epoch_time: StatStream = field(default_factory=lambda: StatStream(drop_first_obs=1))

    step_start: datetime = field(default_factory=datetime.utcnow)
    epoch_start: datetime = field(default_factory=datetime.utcnow)
    step: int = 0
    epoch: int = 0
    total_steps: int = 0
    priority: int = 10

    def guess_batch_size(self, input):
        try:
            if isinstance(input, list):
                return input[0].shape[0]

            if hasattr(input, 'shape'):
                return input.shape[0]

        except Exception:
            return 0

    def on_new_epoch(self, task, epoch, context):
        self.epoch = epoch
        self.epoch_start = datetime.utcnow()

    def on_end_epoch(self, task, epoch, context=None):
        self.epoch_time += get_time_delta(self.epoch_start)

    def on_new_batch(self, task, step, input=None, context=None):
        self.step = step
        self.total_steps += 1
        self.step_start = datetime.utcnow()

    def on_end_batch(self, task, step, input=None, context=None):
        self.step_time += get_time_delta(self.step_start)
        self.batch_size = self.guess_batch_size(input)

    def value(self):
        result = {}

        if self.step_time.count > 0:
            result['step_time'] = self.step_time.avg
            if self.batch_size and self.batch_size > 0:
                result['batch_speed'] = self.batch_size / self.step_time.avg
                result['batch_size'] = self.batch_size

        if self.step_time.count > 2:
            result['step_time_sd'] = self.step_time.sd

        if self.epoch_time.count > 0:
            result['epoch_time'] = self.epoch_time.avg

        if self.epoch_time.count > 2:
            result['epoch_time_sd'] = self.epoch_time.sd

        return result

    def state_dict(self):
        return {
            'speed_step_time': self.step_time.state_dict(),
            'speed_epoch_time': self.step_time.state_dict(),
            'total_steps': self.total_steps,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        self.step_time.from_dict(state_dict['speed_step_time'])
        self.epoch_time.from_dict(state_dict['speed_epoch_time'])

        self.step_start = datetime.utcnow()
        self.epoch_start = datetime.utcnow()

        self.total_steps = state_dict['total_steps']
        self.epoch = state_dict['epoch']


class GuessMaxStep:
    def __init__(self, max_steps=None):
        if max_steps is None:
            self.guessed = False
            self.current_max = float('-inf')
        else:
            self.guessed = True
            self.current_max = max_steps

    def update(self, new_step):
        if new_step is None:
            return

        if new_step > self.current_max - 1:
            self.current_max = new_step + 1
        else:
            self.guessed = True

    def max_step(self):
        if self.guessed:
            return self.current_max

        return 0

    def state_dict(self):
        return dict(guessed=self.guessed, current_max=self.current_max)

    def load_state_dict(self, state_dict):
        self.guessed = state_dict['guessed']
        self.current_max = state_dict['current_max']


def fill(msg, size=40):
    fill_msg = ' ' * (min(0, size - len(msg)))
    return f'{msg}{fill_msg}'


def show_progress(speed: Speed):
    step_time = speed.step_time
    return f'{speed.total_steps:4d} Elapsed time {step_time.total / 60:.2f} min ({step_time.avg:.2f} s/step)'


@dataclass
class DefaultProgress:
    speed: Speed

    def show_progress(self):
        return show_progress(self.speed)

    def state_dict(self):
        return dict()

    def load_state_dict(self, state_dict):
        pass


@dataclass
class EpochProgress:
    speed: Speed
    epochs: int
    steps: GuessMaxStep = field(default_factory=GuessMaxStep)

    def show_progress(self):
        if not self.steps.guessed:
            self.steps.update(self.speed.step)
            return show_progress(self.speed)

        epoch = self.speed.epoch
        step = self.speed.step

        # Compute Total number of steps
        total_steps = self.epochs * self.steps.max_step()
        done_steps = (epoch - 1) * self.steps.max_step() + (step + 1)

        remaining_steps = total_steps - done_steps
        remaining_time = remaining_steps * self.speed.step_time.avg

        if self.speed.step_time.count > 0:
            remaining_time = timedelta(seconds=remaining_time)
        else:
            remaining_time = 'N/A'

        completion = done_steps * 100 / total_steps
        return f'[{completion:6.2f} %] Epoch [{epoch:3d}/{self.epochs:3d}]' \
               f'[{step + 1:4d}/{self.steps.max_step():4d}] ' \
               f'Remaining: {remaining_time}'

    def state_dict(self):
        return dict(steps=self.steps.state_dict())

    def load_state_dict(self, state_dict):
        self.steps.load_state_dict(state_dict['steps'])


@dataclass
class StepProgress:
    speed: Speed
    steps: int

    def show_progress(self):
        total = self.speed.total_steps

        remaining_steps = self.steps - total

        if self.speed.step_time.count > 0:
            remaining_time = timedelta(seconds=remaining_steps * self.speed.step_time.avg)
        else:
            remaining_time = 'N/A'

        return fill(f'[{total:4d}/{self.steps:4d}] Remaining: {remaining_time}')

    def state_dict(self):
        return dict()

    def load_state_dict(self, state_dict):
        pass


class ProgressView(Observer):
    """Print progress regularly

    Parameters
    ----------
    speed: Speed
        speed observer used to gather information about timings
        It is used to compute an estimated end time

    max_epochs: Optional[int]
        The total number of epochs

    max_steps: Optional[int]
        The total number of steps in a single epochs

    Notes
    -----

    If no max epochs nor max steps are specified it outputs
    ``12 Elapsed time 0.12 min (1.00 s/step)``

    if both max epochs and max steps are specified, it outputs
    ``[ 25.00 %] Epoch [  1/  4][  12/  12] Remaining: 0:00:36.042655``

    if only max epochs is specified we will try to guess the max steps during the first epoch.

    """
    def __init__(self, speed: Speed, max_epochs=None, max_steps=None):
        self.print_throttle = option('progress.print.throttle', 30, type=int)
        self.print_fun = print
        self.throttled_print = TimeThrottler(self.print_fun, every=self.print_throttle)

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.speed = speed

        self.progress_printer = DefaultProgress(self.speed)
        self.progress_printer = self.select_progress_printer(max_epochs, max_steps)

        self.frequency_new_epoch: int = 1
        self.frequency_end_epoch: int = option('progress.frequency.epoch', 1, type=int)
        self.frequency_end_batch: int = option('progress.frequency.batch', 1, type=int)
        self.show_metrics: str = option('progress.show.metrics', 'epoch')
        self.frequency_trial: int = 0
        self.worker_id: int = option('worker.id', -1, type=int)
        self.first_epoch = None

    def set_max_epochs(self, epochs):
        self.max_epochs = epochs
        self.select_progress_printer()

    def set_max_steps(self, steps):
        self.max_steps = steps
        self.select_progress_printer()

    def select_progress_printer(self, max_epochs=None, max_steps=None):
        if max_epochs is None:
            max_epochs = self.max_epochs

        if max_steps is None:
            max_steps = self.max_steps

        if max_epochs is not None and max_steps is None:
            self.progress_printer = EpochProgress(self.speed, max_epochs)

        if max_epochs is not None and max_steps is not None:
            self.progress_printer = EpochProgress(self.speed, max_epochs, GuessMaxStep(max_steps))

        if max_epochs is None and max_steps is not None:
            self.progress_printer = StepProgress(self.speed, max_steps)

        return self.progress_printer

    def reset_throttle(self):
        self.throttled_print = TimeThrottler(self.print_fun, every=self.print_throttle)

    def show_progress(self, start='\r', end='\n'):
        worker = ''
        if self.worker_id >= 0:
            worker = f'[W: {self.worker_id:2d}] '

        progress = self.progress_printer.show_progress()
        message = f'{start}{worker}{progress}{end}'

        self.throttled_print(fill(message), end='')

    def on_start_train(self, task, step=None):
        self.print_fun('Starting')

        if task:
            show_dict(task.metrics.value(), print_fun=self.print_fun)

    def on_resume_train(self, task, epoch):
        self.print_fun('Resuming at epoch', epoch)

        if task:
            show_dict(task.metrics.value(), print_fun=self.print_fun)

    def on_end_train(self, task, step=None):
        self.print_fun('Completed training')

        if task:
            show_dict(task.metrics.value())

    def on_new_epoch(self, task, epoch, context):
        if self.first_epoch is None:
            self.first_epoch = epoch

            if epoch == 0:
                warnings.warn('First epoch should 1; epoch 0 is used for the untrained model')

    def on_end_epoch(self, task, epoch, context):
        self.reset_throttle()
        self.show_progress('', '\n')

        if task is not None and self.show_metrics == 'epoch':
            show_dict(task.metrics.value(), print_fun=self.print_fun)

    def on_end_batch(self, task, step, input=None, context=None):
        self.show_progress()

        if task is not None and self.show_metrics == 'batch':
            show_dict(task.metrics.value(), print_fun=self.print_fun)

    def value(self):
        return {}

    def state_dict(self):
        return dict(
            progress_printer=self.progress_printer.state_dict(),
            max_steps=self.max_steps,
            max_epochs=self.max_epochs)

    def load_state_dict(self, state_dict):
        self.progress_printer.load_state_dict(state_dict['progress_printer'])
        self.max_steps = state_dict['max_steps']
        self.max_epochs = state_dict['max_epochs']
        self.select_progress_printer()


@dataclass
class SampleCount(Observer):
    sample_count: int = 0
    epoch: int = 0

    frequency_end_batch: int = 1
    frequency_end_epoch: int = 1

    def state_dict(self):
        return dict(epoch=self.epoch, sample_count=self.sample_count)

    def load_state_dict(self, state_dict):
        self.sample_count = state_dict['sample_count']
        self.epoch = state_dict['epoch']

    def on_end_epoch(self, task, epoch, context):
        self.epoch = epoch

    def on_end_batch(self, task, step, input=None, context=None):
        if hasattr(input, 'shape'):
            batch_size = input.shape[0]
        elif hasattr(input, '__getitem__'):
            batch_size = len(input[0])
        elif input is None:
            batch_size = 1
        else:
            batch_size = input.size(0)

        self.sample_count += batch_size

    def value(self):
        return {
            'sample_count': self.sample_count,
            'epoch': self.epoch
        }


@dataclass
class ElapsedRealTime(Observer):
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime = field(default_factory=datetime.utcnow)

    def state_dict(self):
        return self.value()

    def load_state_dict(self, state_dict):
        self.start_time = self.end_time - timedelta(seconds=state_dict['elapsed_time'])

    def on_end_batch(self, step, task, input=None, context=None):
        self.end_time = datetime.utcnow()

    def on_end_train(self, task, step=None):
        self.end_time = datetime.utcnow()

    @property
    def elapsed_time(self):
        return (self.end_time - self.start_time).total_seconds()

    def value(self):
        return {
            'elapsed_time': self.elapsed_time
        }
