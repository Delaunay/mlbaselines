import torch

from torch.nn import Module, CrossEntropyLoss

from olympus.utils import info
from olympus.tasks.task import Task
from olympus.metrics import OnlineTrainAccuracy, ElapsedRealTime, SampleCount, ProgressView


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

    metrics: MetricListt
        List of metrics to compute for the tasks
    """
    def __init__(self, classifier, optimizer, lr_scheduler, dataloader, criterion=None, device=None,
                 storage=None, logger=None):
        super(Classification, self).__init__(device=device, logger=logger)

        if criterion is None:
            criterion = CrossEntropyLoss()

        self._first_epoch = 0
        self.classifier = classifier
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dataloader = dataloader
        self.criterion = criterion
        self.storage = storage

        self.metrics.append(ElapsedRealTime().every(batch=1))
        self.metrics.append(SampleCount().every(batch=1, epoch=1))
        self.metrics.append(OnlineTrainAccuracy())
        self.metrics.append(ProgressView())

    def get_space(self, **fidelities):
        """Return hyper parameter space"""
        return {
            'task': {       # fidelity(min, max, base logarithm)
                'epochs': fidelities.get('epochs')
            },
            'optimizer': self.optimizer.get_space(),
            'lr_schedule': self.lr_scheduler.get_space(),
            'model': self.model.get_space()
        }

    def init(self, optimizer=None, lr_schedule=None, model=None):
        if optimizer is None:
            optimizer = {}

        if lr_schedule is None:
            lr_schedule = {}

        if model is None:
            model = {}

        self.classifier.init(
            **model
        )
        self.optimizer.init(
            self.classifier.parameters(),
            override=True, **optimizer
        )
        self.lr_scheduler.init(
            self.optimizer,
            override=True, **lr_schedule
        )

        # try to resume itself
        self.resume()

        parameters = {}
        parameters.update(optimizer)
        parameters.update(lr_schedule)
        parameters.update(model)

        self.logger.upsert_trial(parameters)
        self.set_device(self.device)

    def parameters(self):
        return self.classifier.parameters()

    def resume(self):
        state_dict = self.storage.safe_load('checkpoint', device=self.device)

        if not state_dict:
            info('Starting from scratch')
            return False

        try:
            self._first_epoch = state_dict['epoch']
            info(f"Resuming from (epoch: {self._first_epoch})")

            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'], device=self.device)
            self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
            self.dataloader.sampler.load_state_dict(state_dict['sampler'])
            self.metrics.load_state_dict(state_dict['metrics'])

            return True
        except KeyError as e:
            raise KeyError(f'Bad state dictionary!, missing (key: {e.args})') from e

    def checkpoint(self, epoch):
        info(f'Saving checkpoint (epoch: {epoch})')
        was_saved = self.storage.save(
            'checkpoint',
            dict(
                epoch=epoch,
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                lr_scheduler=self.lr_scheduler.state_dict(),
                sampler=self.dataloader.sampler.state_dict(),
                metrics=self.metrics.state_dict()
            )
        )
        if was_saved:
            info('Checkpoint saved')
        else:
            info('Skipped Checkpoint')

    @property
    def metrics(self):
        return self._metrics

    @property
    def model(self) -> Module:
        return self.classifier

    @model.setter
    def model(self, model):
        self.classifier = model

    def resumed(self):
        return self._first_epoch > 0

    def fit(self, epochs, context=None):
        self.classifier.to(self.device)
        progress = self.metrics.get('ProgressView')

        if progress:
            # in case of a resume
            progress.epoch = self._first_epoch
            progress.max_epoch = epochs
            progress.max_step = len(self.dataloader)

        with self.logger as trial_logger:
            if not self.resumed():
                self.metrics.start(self)
                self.report(pprint=True, print_fun=print)
                trial_logger.log_metrics(step=0, **self.metrics.value())

            for epoch in range(self._first_epoch, epochs):
                # Epochs starts from 1 but we iterate from 0 because we are not matlab!
                self.epoch(epoch + 1, context)
                self.report(pprint=True, print_fun=print)

                if epoch != epochs - 1:
                    trial_logger.log_metrics(step=epoch + 1, **self.metrics.value())

            self.metrics.finish(self)
            trial_logger.log_metrics(step=epochs, **self.metrics.value())

    def epoch(self, epoch, context):
        for step, mini_batch in enumerate(self.dataloader):
            self.step(step, mini_batch, context)

        self.metrics.on_new_epoch(epoch, self, context)
        self.lr_scheduler.epoch(epoch, lambda x: self.metrics.value()['validation_accuracy'])
        self.checkpoint(epoch)

    def step(self, step, input, context):
        self.classifier.train()
        self.optimizer.zero_grad()

        batch, target = input
        predictions = self.classifier(batch.to(device=self.device))
        loss = self.criterion(predictions, target.to(device=self.device))

        self.optimizer.backward(loss)
        self.optimizer.step()

        results = {
            # to compute online loss
            'loss': loss.detach(),
            # to compute only accuracy
            'predictions': predictions.detach()
        }

        # Metrics
        self.metrics.on_new_batch(step, self, input, results)
        self.lr_scheduler.step(step)
        return results

    def eval_loss(self, batch):
        self.model.eval()

        with torch.no_grad():
            batch, target = batch
            predictions = self.classifier(batch.to(device=self.device))
            loss = self.criterion(predictions, target.to(device=self.device))

        self.model.train()
        return loss.detach()

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
