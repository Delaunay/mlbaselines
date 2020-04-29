import torch.optim

from olympus.optimizers.schedules.base import LRScheduleAdapter


class WarmUpLR(LRScheduleAdapter):
    def __init__(self, optimizer, warmup_steps, max_steps, iterations='epoch'):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.iterations = iterations

        super(WarmUpLR, self).__init__(
            torch.optim.lr_scheduler.LambdaLR,
            optimizer, lr_lambda=self.lr_lambda
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(
            0.0, float(self.max_steps - step) / float(max(1, self.max_steps - self.warmup_steps))
        )

    def state_dict(self):
        state_dict = self.schedule.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.schedule.load_state_dict(state_dict)

    def epoch(self, epoch=None, metrics=None):
        if self.iterations == 'epoch':
            self.schedule.step()

    def step(self, step=None, metrics=None):
        if self.iterations == 'step':
            self.schedule.step(step)
        print(self.schedule.get_lr())

    @staticmethod
    def get_space():
        return {
            'warmup_steps': 'loguniform(0, 100, discrete=True)',
            'max_steps': 'loguniform(0, 100, discrete=True)',
            'iterations': 'categorical(["epoch", "step"])'}

    @staticmethod
    def defaults():
        return {'warmup_steps': 0, 'max_steps': 120, 'iterations': 'epoch'}


builders = {'warmup': WarmUpLR}
