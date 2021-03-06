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
        return state_dict

    def load_state_dict(self, state_dict):
        self.schedule.load_state_dict(state_dict)

    def epoch(self, epoch=None, metrics=None):
        self.schedule.step()

    def step(self, step=None, metrics=None):
        pass

    @staticmethod
    def get_space():
        return {'gamma': 'loguniform(0.97, 1)'}

    @staticmethod
    def defaults():
        return {'gamma': 0.97}


builders = {'exponential': ExponentialLR}
