import torch
from olympus.metrics import MetricList
from olympus.utils.tracker import BaseTrackLogger
from olympus.utils import BadResume


class BadResumeGuard:
    """Make sure we do not reuse a task with a bad state"""

    def __init__(self, task):
        self.task = task

    def __enter__(self):
        if self.task.bad_state:
            raise BadResume('Cannot resume from bad state! '
                            'You need to create a new task than can resume the previous state')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.task.bad_state = True


class Task:
    def __init__(self, device=None, logger=None):
        self._device = device if device else torch.device('cpu')
        self._logger = logger
        if not self._logger:
            self._logger = BaseTrackLogger()
        self._metrics = MetricList()
        self.bad_state = False

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self.set_device(device)

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger):
        self._logger = logger

    def set_device(self, device):
        for name in dir(self):
            attr = getattr(self, name)

            if hasattr(attr, 'to'):
                setattr(self, name, attr.to(device=device))

        self._device = device

    def eval_loss(self, batch):
        """This is used to compute validation and test loss"""
        raise NotImplementedError()

    def fit(self, epoch, context=None):
        """Execute a single batch

        Parameters
        ----------

        epoch: int
            current step in the training process

        context: dict
            Optional Context

        Notes
        -----
        You should wrap whatever code you have here inside a `BadResumeGuard`
        to prevent users from resuming a failed task that can have a bad states

        To resume a task, you need to create a clean one with the same hyper parameters.
        It will pickup automatically where at its last checkpoint
        """
        raise NotImplementedError()

    @property
    def metrics(self):
        return self._metrics

    def report(self, pprint=True, print_fun=print):
        m = self.metrics
        if m:
            return self.metrics.report(pprint, print_fun)

    def finish(self):
        m = self.metrics
        if m:
            return self.metrics.finish(self)

    def summary(self):
        print(GenerateSummary().task_summary(self))

    def get_space(self, **fidelities):
        """Return missing hyper parameters that need to be set using `init`"""
        raise NotImplementedError()

    def init(self, **kwargs):
        """Used to initialize the hyperparameters is any"""
        raise NotImplementedError()

    def resume(self):
        """Try to load a previous unfinished state to resume

        Notes
        -----
        You should wrap whatever code you have here inside a `BadResumeGuard`
        to prevent users from resuming a failed task that can have a bad states

        To resume a task, you need to create a clean one with the same hyper parameters.
        It will pickup automatically where at its last checkpoint
        """
        raise NotImplementedError()

    def checkpoint(self, step):
        """Save a state the task can go back to if an error occur"""
        raise NotImplementedError()


class GenerateSummary:
    dispatch = {
        'Model': lambda model: model.model,
        'Optimizer': lambda optimizer: optimizer.optimizer,
        'DataLoader': lambda data: data.dataset,
        'TransformedSubset': lambda data: data.dataset,
        'MetricList': lambda metrics: metrics.metrics,
        'LRSchedule': lambda schedule: schedule.lr_scheduler
    }

    _rename = {
        '_metrics': 'metrics',
        '_device': 'device',
        '_first_epoch': 'first_epoch'
    }

    def __init__(self):
        self.output = []

    def print(self, msg='', end='\n'):
        self.output.append(f'{msg}{end}')

    def is_nested(self, name):
        return name in GenerateSummary.dispatch

    def retrieve_nested(self, name, obj):
        return GenerateSummary.dispatch.get(name, lambda x: x)(obj)

    def rename(self, name):
        return GenerateSummary._rename.get(name, name)

    def get_name(self, attr, obj, type_name, depth=0):
        self.print(f'{"  " * depth} {self.rename(attr)}: ', end='')

        if not self.is_nested(type_name):
            if type_name == 'device':
                self.print(str(obj))
            elif type_name == 'list':
                self.print()
                for item in obj:
                    self.print(f'{"  " * (depth + 1)} - {type(item).__name__}')
            else:
                self.print(type_name)

        else:
            self.print()
            nested = self.retrieve_nested(type_name, obj)
            nested_type = type(nested).__name__
            self.get_name(type_name, nested, nested_type, depth + 1)

    def task_summary(self, obj):
        self.output = []

        self.print('=' * 80)
        self.print(type(obj).__name__)
        self.print('-' * 80)
        for attr, value in obj.__dict__.items():
            type_name = type(value).__name__
            self.get_name(attr, value, type_name)
        self.print('=' * 80)

        return ''.join(self.output)
