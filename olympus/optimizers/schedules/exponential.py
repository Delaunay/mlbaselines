import torch.optim

from olympus.optimizers.schedules.base import ScheduleBuilder, LRSchedule


class ExponentialScheduleBuilder(ScheduleBuilder):
    def build(self, optimizer, gamma):
        return ExponentialLR(torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma))

    def get_space(self):
        return {'gamma': 'loguniform(0.97, 1)'}


class ExponentialLR(LRSchedule):
    def epoch(self, epoch, metrics=None):
        self.lr_scheduler.step(epoch=epoch)

    def step(self, step, metrics=None):
        pass


builders = {'exponential': ExponentialScheduleBuilder}
