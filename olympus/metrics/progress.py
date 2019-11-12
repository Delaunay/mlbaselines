from dataclasses import dataclass, field


from olympus.metrics.metric import Metric
from olympus.utils.stat import StatStream
from olympus.utils.options import option

from datetime import datetime, timedelta


def get_time_delta(start):
    return (datetime.utcnow() - start).total_seconds()


@dataclass
class ProgressView(Metric):
    print_fun = print
    epoch = 0
    step = 0
    max_epoch: int = 0
    max_step: int = 0
    batch_size: int = 0
    step_length: int = 0

    frequency_epoch: int = option('progress.frequency_epoch', 1, type=int)
    frequency_batch: int = option('progress.frequency_batch', 1, type=int)
    frequency_trial: int = 0

    step_time: StatStream = field(default_factory=lambda: StatStream(drop_first_obs=5))
    epoch_time: StatStream = field(default_factory=lambda: StatStream(drop_first_obs=1))

    step_start: datetime = field(default_factory=datetime.utcnow)
    epoch_start: datetime = field(default_factory=datetime.utcnow)

    def show_progress(self, epoch, step=None):
        if step is None:
            step = ' ' * self.step_length
        else:
            step = f'Step [{step:3d}/{self.max_step:3d}]'
            self.step_length = len(step)

        self.print_fun(f'\rEpoch [{epoch:3d}/{self.max_epoch:3d}] {step} {self.eta(epoch)}', end='')

    def eta(self, epoch):
        if self.step_time.count > 0:
            total_steps = self.max_step * self.max_epoch
            spent_steps = self.max_step * epoch + self.step
            remaining_steps = total_steps - spent_steps

            avg = self.step_time.avg
            # if we spent enough epochs estimate using both duration
            if self.epoch_time.count > 0:
                avg = (avg + self.epoch_time.avg / float(self.max_step)) / 2

            step_estimate = avg * remaining_steps
            return f'ETA: {step_estimate / 60:9.4f} min'

        return ''

    def on_new_epoch(self, epoch, task, context):
        self.epoch_time += get_time_delta(self.epoch_start)
        self.epoch_start = datetime.utcnow()

        self.epoch = epoch
        self.step = 0
        print()

        self.max_epoch = max(self.epoch, self.max_epoch)
        self.show_progress(epoch)
        print()

    def guess_batch_size(self, input):
        try:
            return input[0].shape[0]
        except Exception:
            return 0

    def on_new_batch(self, step, task, input, context):
        self.step_time += get_time_delta(self.step_start)
        self.step_start = datetime.utcnow()
        self.batch_size = self.guess_batch_size(input)

        self.step = step
        self.max_step = max(self.step, self.max_step)

        self.show_progress(self.epoch, step=self.step)

    def start(self, task=None):
        self.step_start = datetime.utcnow()
        self.epoch_start = self.step_start

    def finish(self, task=None):
        print()

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
        return dict(
            max_epoch=self.max_epoch,
            max_step=self.max_step
        )

    def load_state_dict(self, state_dict):
        self.max_epoch = state_dict['max_epoch']
        self.max_step = state_dict['max_step']
        self.step = 0
        self.epoch = 0


@dataclass
class SampleCount(Metric):
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
            batch_size = input[0].size(0)
        else:
            batch_size = input.size(0)

        self.sample_count += batch_size

    def value(self):
        return {
            'sample_count': self.sample_count,
            'epoch': self.epoch
        }


@dataclass
class ElapsedRealTime(Metric):
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


