import torch.optim

from olympus.optimizers.schedules.base import LRScheduleAdapter


class CyclicLR(LRScheduleAdapter):
    def __init__(self, optimizer, base_lr, max_lr, step_size_up, step_size_down, mode, base_momentum, max_momentum):
        super(CyclicLR, self).__init__(
            torch.optim.lr_scheduler.CyclicLR,
            optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up,
            step_size_down=step_size_down, mode=mode, base_momentum=base_momentum,
            max_momentum=max_momentum
        )

    def state_dict(self):
        state_dict = self.schedule.state_dict()
        state_dict.pop('scale_fn')
        return state_dict

    def load_state_dict(self, state_dict):
        state_dict['scale_fn'] = self.schedule.scale_fn
        self.schedule.load_state_dict(state_dict)

    def epoch(self, epoch=None, metrics=None):
        pass

    def step(self, step=None, metrics=None):
        self.schedule.step()

    @staticmethod
    def get_space():
        return {
            'base_lr': 'loguniform(1e-5, 1e-2)',
            'max_lr': 'loguniform(1e-2, 1)',
            'step_size_up': 'loguniform(1000, 50000)',
            'step_size_down': 'loguniform(1000, 50000)',
            'mode': 'choices(["triangular", "triangular2", "exp_range"])',
            'base_momentum': 'uniform(0.7, 0.9)',
            'max_momentum': 'loguniform(0.9, 0.99)',
        }

    @staticmethod
    def defaults():
        return {
            'base_lr': 1e-5,
            'max_lr': 1e-2,
            'step_size_up': 2000,
            'step_size_down': 1000,
            'mode': 'triangular',
            'base_momentum': 0.7,
            'max_momentum': 0.99,
        }


builders = {'cyclic': CyclicLR}
