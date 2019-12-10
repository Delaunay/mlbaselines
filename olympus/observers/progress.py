from dataclasses import dataclass, field

from olympus.observers.observer import Observer
from olympus.utils.stat import StatStream
from olympus.utils.options import option

from datetime import datetime, timedelta
from typing import Optional


def get_time_delta(start):
    return (datetime.utcnow() - start).total_seconds()


@dataclass
class Speed(Observer):
    epoch = 0
    step = 0

    batch_size: int = 0

    frequency_epoch: int = 1
    frequency_batch: int = 1

    step_time: StatStream = field(default_factory=lambda: StatStream(drop_first_obs=5))
    epoch_time: StatStream = field(default_factory=lambda: StatStream(drop_first_obs=1))

    step_start: datetime = field(default_factory=datetime.utcnow)
    epoch_start: datetime = field(default_factory=datetime.utcnow)

    def guess_batch_size(self, input):
        try:
            return input[0].shape[0]
        except Exception:
            return 0

    def on_new_epoch(self, epoch, task, context):
        self.epoch_time += get_time_delta(self.epoch_start)
        self.epoch_start = datetime.utcnow()

        self.epoch = epoch
        self.step = 0

    def on_new_batch(self, step, task, input, context):
        self.batch_size = self.guess_batch_size(input)
        self.step_time += get_time_delta(self.step_start)
        self.step_start = datetime.utcnow()
        self.step = step

    def start(self, task=None):
        self.step_start = datetime.utcnow()
        self.epoch_start = self.step_start

    def finish(self, task=None):
        pass

    def value(self):
        result = {}

        if self.step_time.count > 0:
            result['step_time'] = self.step_time.avg
            if self.batch_size > 0:
                result['batch_speed'] = self.batch_size / self.step_time.avg

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
            'speed_epoch': self.epoch,
        }

    def load_state_dict(self, state_dict):
        self.step_time.from_dict(state_dict['speed_step_time'])
        self.epoch_time.from_dict(state_dict['speed_epoch_time'])
        self.epoch = state_dict['speed_epoch']

        self.step_start = datetime.utcnow()
        self.epoch_start = datetime.utcnow()
        self.step = 0


@dataclass
class ProgressView(Observer):
    speed_observer: Optional[Speed] = None

    print_fun = print
    max_epoch: int = 0
    max_step: int = 0
    step_length: int = 0
    epoch: int = 0
    multiplier: int = 0

    frequency_epoch: int = option('progress.frequency_epoch', 1, type=int)
    frequency_batch: int = option('progress.frequency_batch', 1, type=int)
    frequency_trial: int = 0

    orion_handle = None
    worker_id: int = option('worker.id', -1, type=int)

    def show_progress(self, epoch, step=None):
        if step is None:
            step = ' ' * self.step_length
        else:
            step = f'Step [{step:3d}/{self.max_step:3d}]'
            self.step_length = len(step)

        hpo = ''
        if self.orion_handle is not None:
            hpo_completion = self.overall_progress()
            hpo = f'HPO [{hpo_completion:6.2f}%] '

        worker = ''
        if self.worker_id >= 0:
            worker = f'[W: {self.worker_id:2d}] '

        eta = ''
        if self.speed_observer:
            eta = self.eta(self.speed_observer, epoch)

        self.print_fun(
            f'\r{worker}{hpo}Epoch [{epoch:3d}/{self.max_epoch:3d}] {step} {eta}', end='')

    def overall_progress(self):
        """Return the overall HPO progress in % completion"""
        return len(self.orion_handle.fetch_trials_by_status('completed')) * 100 / self.number_of_trials()

    def number_of_trials(self):
        # FIXME: Get max trials for the algo itself
        return self.orion_handle.max_trials

    def estimate_time_trial_finish(self, obs, epoch):
        """Estimate when a trial will finish"""
        if obs.step_time.count == 0:
            return None

        total_steps = self.max_step * self.max_epoch
        spent_steps = self.max_step * epoch + obs.step
        remaining_steps = total_steps - spent_steps

        avg = obs.step_time.avg
        # if we spent enough epochs estimate using both duration
        if obs.epoch_time.count > 0:
            avg = (avg + obs.epoch_time.avg / float(self.max_step)) / 2

        step_estimate = avg * remaining_steps
        return step_estimate

    def eta(self, obs, epoch):
        step_estimate = self.estimate_time_trial_finish(obs, epoch)

        if step_estimate:
            return f'ETA: {step_estimate / 60:9.4f} min'

        return ''

    def on_new_epoch(self, epoch, task, context):
        self.epoch = epoch

        if self.speed_observer:
            self.max_epoch = max(self.speed_observer.epoch, self.max_epoch)

        self.print_fun()
        self.show_progress(epoch)
        self.print_fun()

    def on_new_batch(self, step, task, input, context):
        if self.speed_observer:
            step = self.speed_observer.step
            self.max_step = max(step, self.max_step)
            self.epoch = self.speed_observer.epoch

        self.show_progress(self.epoch, step)

    def start(self, task=None):
        pass

    def finish(self, task=None):
        self.print_fun()

    def init_speed_observer(self, task):
        if not self.speed_observer and task:
            self.speed_observer = task.metrics.get('Speed', None)

    def value(self):
        return {}

    def state_dict(self):
        return dict(
            max_epoch=self.max_epoch,
            max_step=self.max_step
        )

    def load_state_dict(self, state_dict):
        self.max_epoch = state_dict['max_epoch']
        self.max_step = state_dict['max_step']


@dataclass
class SampleCount(Observer):
    sample_count: int = 0
    epoch: int = 0

    def state_dict(self):
        return dict(epoch=self.epoch, sample_count=self.sample_count)

    def load_state_dict(self, state_dict):
        self.sample_count = state_dict['sample_count']
        self.epoch = state_dict['epoch']

    def on_new_epoch(self, epoch, task, context):
        self.epoch = epoch

    def on_new_batch(self, step, task, input, context):
        if hasattr(input, '__getitem__'):
            batch_size = len(input[0])
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

    def start(self, task=None):
        pass

    def state_dict(self):
        return self.value()

    def load_state_dict(self, state_dict):
        self.start_time = self.end_time - timedelta(seconds=state_dict['elapsed_time'])

    def on_new_batch(self, step, task, input, context):
        self.end_time = datetime.utcnow()

    def finish(self, task=None):
        self.end_time = datetime.utcnow()

    @property
    def elapsed_time(self):
        return (self.end_time - self.start_time).total_seconds()

    def value(self):
        return {
            'elapsed_time': self.elapsed_time
        }
