from olympus.optimizers.schedules.base import LRScheduleI


class NoLR(LRScheduleI):
    def __init__(self, optimizer):
        super(NoLR, self).__init__(optimizer)

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def epoch(self, epoch, metrics=None):
        pass

    def step(self, step, metrics=None):
        pass

    @staticmethod
    def get_space():
        return {}


builders = {'none': NoLR}
