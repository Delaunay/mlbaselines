import torch


class Task:
    def __init__(self, device=None):
        self._device = device if device else torch.device('cpu')

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        for name in dir(self):
            attr = getattr(self, name)
            if hasattr(attr, 'to'):
                setattr(self, name, attr.to(device=device))

        self._device = device

    def fit(self, step, input, context):
        """Execute a single batch

        Parameters
        ----------

        step: int
            current batch_id

        input: Tensor. Tuple[Tensor]
            Pytorch tensor or tuples of tensors

        context: dict
            Optional Context
        """
        raise NotImplementedError()

    @property
    def metrics(self):
        pass

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
