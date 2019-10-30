from dataclasses import dataclass
import json

import torch

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer

from olympus.tasks.task import Task
from olympus.metrics import TrainAccuracy, ElapsedRealTime, SampleCount, ClassifierAdversary, MetricList


@dataclass
class Classification(Task):
    classifier: Module
    optimizer: Optimizer
    criterion: Module = CrossEntropyLoss()
    _metrics = MetricList(
        TrainAccuracy().every(epoch=1, batch=1),
        ElapsedRealTime().every(batch=1),
        SampleCount().every(batch=1, epoch=1),
        ClassifierAdversary(epsilon=0.25).every(epoch=1, batch=1)
    )

    @property
    def metrics(self):
        return self._metrics

    @property
    def model(self) -> Module:
        return self.classifier

    @model.setter
    def model(self, model):
        self.classifier = model

    def fit(self, dataloader, epochs, context):
        steps = 0
        for epoch in range(epochs):
            print(epoch)
            steps = self.epoch(dataloader, epoch, steps, context)

    def epoch(self, dataloader, epoch, steps, context):
        for mini_batch in dataloader:
            print(steps)
            self.step(steps, mini_batch, context)
            steps += 1
        
        return steps

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
