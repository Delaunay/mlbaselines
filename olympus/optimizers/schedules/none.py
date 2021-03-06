from olympus.optimizers.schedules.base import LRScheduleInterface


class NoLR(LRScheduleInterface):
    def __init__(self, optimizer):
        super(NoLR, self).__init__(optimizer)

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def epoch(self, epoch=None, metrics=None):
        pass

    def step(self, step=None, metrics=None):
        pass

    @staticmethod
    def get_space():
        return {}


builders = {'none': NoLR}
