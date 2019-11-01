import json
import logging

import torch

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer

from olympus.tasks.task import Task
from olympus.metrics import OnlineTrainAccuracy, ElapsedRealTime, SampleCount, ClassifierAdversary, MetricList


logging.getLogger(__name__)


class Classification(Task):
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

        loss.backward()
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
