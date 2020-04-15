from torch.optim.lr_scheduler import _LRScheduler


class LRScheduleInterface(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(LRScheduleInterface, self).__init__(optimizer, last_epoch)

    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self, state_dict):
        raise NotImplementedError()

    def epoch(self, epoch=None, metrics=None):
        super(LRScheduleInterface, self).step(epoch)

    def step(self, step=None, metrics=None):
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

    def epoch(self, epoch=None, metrics=None):
        return self.schedule.step(epoch)

    def step(self, step=None, metrics=None):
        pass

    def get_lr(self):
        # pytorch 1.4: get_lr becomes get_last_lr
        # Scheduler.get_lr is also used internally for computing new learning rates,"
        # this actually returns a value that is “one step ahead.”
        # optimizer.param_groups[0]['lr'] was in version 1.3.1 and remains
        # in 1.4.0 a way of getting the current learning rate used in the optimizer.
        try:
            return self.schedule.get_lr()
        except:
            return self.schedule.get_last_lr()

    @staticmethod
    def get_space():
        raise NotImplementedError()

    @staticmethod
    def defaults():
        raise NotImplementedError()
