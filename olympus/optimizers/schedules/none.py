from olympus.optimizers.schedules.base import ScheduleBuilder, LRSchedule


class NoScheduleBuilder(ScheduleBuilder):
    def build(self, optimizer):
        return NoLR()

    def get_space(self):
        return {}


class NoLR(LRSchedule):
    def __init__(self):
        pass

    def state_dict(self):
        return {}
    
    def load_state_dict(self, state_dict):
        pass

    def epoch(self, epoch, metrics=None):
        pass

    def step(self, step, metrics=None):
        pass


builders = {'none': NoScheduleBuilder}
