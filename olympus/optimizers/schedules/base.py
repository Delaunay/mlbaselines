class ScheduleBuilder:
    def __call__(self, optimizer, **kwargs):
        return self.build(optimizer, **kwargs)

    def build(self, optimizer, **kwargs):
        raise NotImplementedError

    def get_space(self):
        raise NotImplementedError

    def get_params(self, params):
        optimizer_params = dict()

        for key in self.get_space().keys():
            optimizer_params[key] = params[key]

        return optimizer_params


class LRSchedule:

    def __init__(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict)

    def epoch(self, epoch, metrics=None):
        pass

    def step(self, step, metrics=None):
        pass

    def get_lr(self):
        return self.lr_scheduler.get_lr()
