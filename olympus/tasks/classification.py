import torch

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer

from olympus.tasks.task import Task
from olympus.metrics import OnlineTrainAccuracy, ElapsedRealTime, SampleCount, ClassifierAdversary, MetricList


class Classification(Task):
    """Train a model to recognize a range of classes

    Attributes
    ----------
    classifier: Module
        Module taking sample data and returning the probability of the sample belonging to a range of classes

    optimizer: Optimizer
        Optimizer taking model's parameters

    criterion: Module
        Function evaluating the quality of the model's predictions, also named cost function or loss function

    dataloader: Iterator
        Batch sample iterator used to train the model

    device:
        Acceleration device to run the task on

    storage: Storage
        Where to save checkpoints in case of failures

    metrics: MetricList
        List of metrics to compute for the tasks
    """
    def __init__(self, classifier, optimizer, lr_scheduler, dataloader, criterion=None, device=None,
                 storage=None):
        super(Classification, self).__init__(device=device)

        if criterion is None:
            criterion = CrossEntropyLoss()

        self._first_epoch = 1
        self.classifier = classifier
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dataloader = dataloader
        self.criterion = criterion
        self.storage = storage

        self._metrics = MetricList(
            ElapsedRealTime().every(batch=1),
            SampleCount().every(batch=1, epoch=1),
            OnlineTrainAccuracy(),
            # ClassifierAdversary(epsilon=0.25).every(epoch=1, batch=1)
        )

    def resume(self):
        try:
            state_dict = self.storage.load('checkpoint')
        except RuntimeError as e:
            if 'CPU-only machine' in str(e):
                raise KeyboardInterrupt('Job got scheduled on bad node.') from e
        except FileNotFoundError:
            print('Starting from scratch')
            return

        print(f"Resuming from epoch {state_dict['epoch']}")
        self._first_epoch = state_dict['epoch'] + 1
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        # Dirty fix found here:
        # https://github.com/pytorch/pytorch/issues/2830#issuecomment-336194949
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.dataloader.sampler.load_state_dict(state_dict['sampler'])
        self.metrics.load_state_dict(state_dict['metrics'])

    def checkpoint(self, epoch):
        self.storage.save(
            'checkpoint',
            dict(
                epoch=epoch,
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                lr_scheduler=self.lr_scheduler.state_dict(),
                sampler=self.dataloader.sampler.state_dict(),
                metrics=self.metrics.state_dict()
                ))

    @property
    def metrics(self):
        return self._metrics

    @property
    def model(self) -> Module:
        return self.classifier

    @model.setter
    def model(self, model):
        self.classifier = model

    def fit(self, epochs, context):
        self.classifier.to(self.device)

        if self._first_epoch == 1:
            print('\rEpoch   0: ', end='')
            self.metrics.epoch(0, self, context)
            self.report(pprint=True, print_fun=print)

        for epoch in range(self._first_epoch, epochs + 1):
            print(f'\rEpoch {epoch:3d}: ', end='')
            self.epoch(epoch, context)
            self.report(pprint=True, print_fun=print)

        print()

    def epoch(self, epoch, context):
        for step, mini_batch in enumerate(self.dataloader):
            self.step(step, mini_batch, context)

        self.metrics.epoch(epoch, self, context)
        self.lr_scheduler.epoch(epoch, self.metrics.value()['validation_accuracy'])
        self.checkpoint(epoch)
        print(self.lr_scheduler.get_lr()[0])

    def step(self, step, input, context):
        self.classifier.train()
        self.optimizer.zero_grad()

        batch, target = input
        predictions = self.classifier(batch.to(device=self.device))
        loss = self.criterion(predictions, target.to(device=self.device))

        self.optimizer.backward(loss)
        self.optimizer.step()

        results = {
            'loss': loss.detach(),
            'predictions': predictions.detach()
        }

        # Metrics
        self.metrics.step(step, self, input, results)
        self.lr_scheduler.step(step)
        return results

    def predict_probabilities(self, batch):
        with torch.no_grad():
            self.classifier.eval()
            return self.classifier(batch.to(device=self.device))

    def predict(self, batch, target=None):
        probabilities = self.predict_probabilities(batch)
        _, predicted = torch.max(probabilities, 1)

        loss = None
        if target is not None:
            loss = self.criterion(probabilities, target.to(device=self.device))

        return predicted, loss

    def accuracy(self, batch, target):
        predicted, loss = self.predict(batch, target)
        acc = (predicted == target.to(device=self.device)).sum()

        return acc.float() / target.size(0), loss

    def summary(self):
        GenerateSummary().task_summary(self)


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

    def is_nested(self, name):
        return name in GenerateSummary.dispatch

    def retrieve_nested(self, name, obj):
        return GenerateSummary.dispatch.get(name, lambda x: x)(obj)

    def rename(self, name):
        return GenerateSummary._rename.get(name, name)

    def get_name(self, attr, obj, type_name, depth=0):
        print(f'{"  " * depth} {self.rename(attr)}: ', end='')

        if not self.is_nested(type_name):
            if type_name == 'device':
                print(str(obj))
            elif type_name == 'list':
                print()
                for item in obj:
                    print(f'{"  " * (depth + 1)} - {type(item).__name__}')
            else:
                print(type_name)

        else:
            print()
            nested = self.retrieve_nested(type_name, obj)
            nested_type = type(nested).__name__
            self.get_name(type_name, nested, nested_type, depth + 1)

    def task_summary(self, obj):
        print('=' * 80)
        print(type(obj).__name__)
        print('-' * 80)
        for attr, value in obj.__dict__.items():
            type_name = type(value).__name__
            self.get_name(attr, value, type_name)
        print('=' * 80)

