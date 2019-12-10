from torch.optim.lr_scheduler import _LRScheduler


class LRScheduleInterface(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(LRScheduleInterface, self).__init__(optimizer, last_epoch)

    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self, state_dict):
        raise NotImplementedError()

    def epoch(self, epoch, metrics=None):
        super(LRScheduleInterface, self).step(epoch)

    def step(self, step, metrics=None):
        pass

    def get_lr(self):
        raise NotImplementedError()

    @staticmethod
    def get_space():
        raise NotImplementedError()

    @staticmethod
    def defaults():
        return {}


class LRScheduleAdapter(LRScheduleInterface):
    def __init__(self, builder, optimizer, last_epoch=-1, **kwargs):
        self.schedule = builder(optimizer=optimizer, last_epoch=last_epoch, **kwargs)

    def state_dict(self):
        return self.schedule.state_dict()

    def load_state_dict(self, state_dict):
        return self.schedule.load_state_dict(state_dict)

    def epoch(self, epoch, metrics=None):
        return self.schedule.step(epoch)

    def step(self, step, metrics=None):
        pass

    def get_lr(self):
        return self.schedule.get_lr()

    @staticmethod
    def get_space():
        raise NotImplementedError()

    @staticmethod
    def defaults():
        raise NotImplementedError()
