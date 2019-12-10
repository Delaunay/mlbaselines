import torch

from olympus.optimizers import Optimizer
from olympus.models import Model
from olympus.optimizers.schedules import LRSchedule
from olympus.optimizers.schedules.base import LRScheduleInterface


class MyExponentialLR(LRScheduleInterface):
    def __init__(self, optimizer, gamma):
        super(MyExponentialLR, self).__init__(optimizer)
        self.gamma = gamma

    def state_dict(self):
        state = super(MyExponentialLR, self).state_dict()
        state['gamma'] = self.gamma
        return state

    def load_state_dict(self, state_dict):
        self.gamma = state_dict.pop('gamma')
        return super(MyExponentialLR, self).load_state_dict(state_dict)

    def epoch(self, epoch, metrics=None):
        self._step_count += 1

        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        return [base_lr * self.gamma ** self.last_epoch
                for base_lr in self.base_lrs]

    @staticmethod
    def get_space():
        return {'gamma': 'loguniform(0.97, 1)'}


if __name__ == '__main__':
    model = Model(
        'logreg',
        input_size=(290,),
        output_size=(10,)
    )

    optimizer = Optimizer('sgd', params=model.parameters())

    # If you use an hyper parameter optimizer, it will generate this for you
    optimizer.init(lr=1e-4, momentum=0.02, weight_decay=1e-3)

    schedule = LRSchedule(schedule=MyExponentialLR)
    schedule.init(optimizer=optimizer, gamma=0.97)

    optimizer.zero_grad()

    input = torch.randn((10, 290))
    output = model(input)
    loss = output.sum()
    loss.backward()

    optimizer.step()

    print(optimizer.param_groups[0]['lr'])
    schedule.epoch(1)
    print(optimizer.param_groups[0]['lr'])


