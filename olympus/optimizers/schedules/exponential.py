import torch.optim

from olympus.optimizers.schedules.base import LRScheduleAdapter


class ExponentialLR(LRScheduleAdapter):
    def __init__(self, optimizer, gamma):
        super(ExponentialLR, self).__init__(
            torch.optim.lr_scheduler.ExponentialLR,
            optimizer, gamma=gamma
        )

    def state_dict(self):
        state_dict = self.schedule.state_dict()
        state_dict.pop('scale_fn')
        return state_dict

    def load_state_dict(self, state_dict):
        state_dict['scale_fn'] = self.schedule.scale_fn
        self.schedule.load_state_dict(state_dict)

    def epoch(self, epoch, metrics=None):
        self.schedule.step(epoch)

    def step(self, step, metrics=None):
        pass

    @staticmethod
    def get_space():
        return {'gamma': 'loguniform(0.97, 1)'}


builders = {'exponential': ExponentialLR}
