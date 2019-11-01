import torch.optim

from olympus.optimizers.schedules.base import ScheduleBuilder, LRSchedule


class CyclicScheduleBuilder(ScheduleBuilder):
    def build(self, optimizer, base_lr, max_lr, step_size_up, step_size_down, mode, base_momentum,
              max_momentum):
        return CyclicLR(
            torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up,
                step_size_down=step_size_down, mode=mode, base_momentum=base_momentum,
                max_momentum=max_momentum))

    def get_space(self):
        return {'base_lr': 'loguniform(1e-5, 1e-2)',
                'max_lr': 'loguniform(1e-2, 1)',
                'step_size_up': 'loguniform(1000, 50000)',
                'step_size_down': 'loguniform(1000, 50000)',
                'mode': 'choices(["triangular", "triangular2", "exp_range"])',
                'base_momentum': 'uniform(0.7, 0.9)',
                'max_momentum': 'loguniform(0.9, 0.99)',
                }


class CyclicLR(LRSchedule):

    def state_dict(self):
        state_dict = self.lr_scheduler.state_dict()
        state_dict.pop('scale_fn')
        return state_dict

    def load_state_dict(self, state_dict):
        state_dict['scale_fn'] = self.lr_scheduler.scale_fn
        self.lr_scheduler.load_state_dict(state_dict)

    def epoch(self, epoch, metrics=None):
        pass

    def step(self, step, metrics=None):
        self.lr_scheduler.step()


builders = {'cyclic': CyclicScheduleBuilder}
