from dataclasses import dataclass
import datetime
import json

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable


@dataclass
class Metric:
    frequency_epoch: int = 0
    frequency_batch: int = 0

    def on_new_epoch(self, step, task, input, context):
        pass

    def on_new_batch(self, step, task, input, context):
        pass

    def finish(self, task):
        pass

    def value(self):
        return dict()

    def every(self, *args, epoch=None, batch=None):
        assert len(args) == 0

        if epoch is not None:
            self.frequency_epoch = epoch

        if batch is not None:
            self.frequency_batch = batch

        return self


class MetricList:
    def __init__(self, *args):
        self.metrics = list(args)
        self.batch_id: int = 0
        self.epoch: int = 0
        self._previous_step = 0
        self.name_2_metrics = {}

    def __getitem__(self, item, default=None):
        return self.name_2_metrics.get(item, default)

    def metric_keys(self):
        return self.name_2_metrics.keys()

    def append(self, m: Metric):
        self.metrics.append(m)
        self.name_2_metrics[m.__class__] = m

    def step(self, step, task, input, context):
        if self._previous_step > step:
            for m in self.metrics:
                if m.frequency_epoch > 0 and self.epoch % m.frequency_epoch == 0:
                    m.on_new_epoch(step, task, input, context)

            self.epoch += 1
            self.batch_id = 0

        for m in self.metrics:
            if m.frequency_batch > 0 and self.batch_id % m.frequency_batch == 0:
                m.on_new_batch(step, task, input, context)

        self.batch_id += 1
        self._previous_step = step

    def finish(self, task):
        for m in self.metrics:
            m.finish(task)

    def value(self):
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.value())

        return metrics

    def report(self, pprint=True, print_fun=print):
        metrics = self.value()

        if pprint:
            print_fun(json.dumps(metrics, indent=2))

        return metrics


class MeanAccumulator:
    n = 0
    acc = 0

    def update(self, v):
        self.acc += v
        self.n += 1

    def compute(self):
        mean = self.acc / self.n
        self.n = 0
        self.acc = 0
        return mean


@dataclass
class ValidationAccuracy(Metric):
    loader: DataLoader = None
    accuracies = []
    losses = []
    frequency = 1

    def on_new_epoch(self, step, task, input, context):
        acc = 0
        loss_acc = 0

        count = len(self.loader)
        for input in self.loader:
            accuracy, loss = task.accuracy(input[0], input[1])

            acc += accuracy.item()
            loss_acc += loss.item()

        self.accuracies.append(acc / count)
        self.losses.append(loss_acc / count)

    def finish(self, task):
        self.on_new_epoch(None, task, None, None)

    def value(self):
        return {
            'validation_accuracy': self.accuracies[-1],
            'validation_loss': self.losses[-1]
        }


@dataclass
class TrainAccuracy(Metric):
    accuracies = []
    losses = []
    accumulator = 0
    loss = 0
    count = 0

    def on_new_batch(self, step, task, input, context):
        _, targets = input
        predictions = context.get('predictions')

        # compute accuracy for the current batch
        if predictions is not None:
            _, predicted = torch.max(predictions, 1)

            target = input[1].to(device=task.device)

            loss = task.criterion(predictions, target).item()
            acc = (predicted == target).sum().item() / target.size(0)

            self.accumulator += acc
            self.loss += loss
            self.count += 1

    def on_new_epoch(self, step, task, input, context):
        # new epoch
        self.accuracies.append(self.accumulator / self.count)
        self.losses.append(self.loss / self.count)
        self.accumulator = 0
        self.loss = 0
        self.count = 0

    def finish(self, task):
        if self.count > 0:
            self.on_new_epoch(None, None, None, None)

    def value(self):
        return {
            'train_accuracy': self.accuracies[-1],
            'train_loss': self.losses[-1]
        }


@dataclass
class NamedMetric(Metric):
    name: str = None
    metrics = []
    accumulator = 0
    count = 0

    def on_new_epoch(self, step, task, input, context):
        self.metrics.append(self.accumulator / self.count)
        self.accumulator = 0
        self.count = 0

    def on_new_batch(self, step, task, input, context):
        value = context.get(self.name)
        if value is not None:
            self.accumulator += value
            self.count += 1

    def finish(self, task):
        if self.count > 0:
            self.on_new_epoch(None, None, None, None)

    def value(self):
        return {
            self.name: self.metrics[-1]
        }


@dataclass
class SampleCount(Metric):
    sample_count: int = 0
    epoch: int = 0

    def on_new_epoch(self, step, task, input, context):
        self.epoch += 1

    def on_new_batch(self, step, task, input, context):
        if hasattr(input, '__getitem__'):
            batch_size = input[0].size(0)
        else:
            batch_size = input.size(0)

        self.sample_count += batch_size

    def value(self):
        return {
            'sample_count': self.sample_count,
            'epoch': self.epoch
        }


@dataclass
class ElapsedRealTime(Metric):
    start = datetime.datetime.utcnow()
    end = datetime.datetime.utcnow()

    def on_new_batch(self, step, task, input, context):
        self.end = datetime.datetime.utcnow()

    @property
    def elapsed_time(self):
        return (self.end - self.start).seconds

    def value(self):
        return {
            'elapsed_time': self.elapsed_time
        }


@dataclass
class ClassifierAdversary(Metric):
    """Simple Adversary Generator from https://arxiv.org/pdf/1412.6572.pdf.
    Measure how robust a network is from adversary attacks

        image = original_image + epsilon * sign(grad(cost(theta, original_image, t), original_image)

    epsilon corresponds to the magnitude of the smallest bit of an image encoding converted to real number

    ImageNet: 0.07
    MNIST: 0.25

    """
    epsilon: float = 0.25
    accuracies = []
    losses = []
    distortions = []
    distortion = 0
    loss = 0
    accumulator = 0
    count = 0
    loader: DataLoader = None

    def on_new_epoch(self, step, task, input, context):
        if self.loader:
            accuracy = 0
            total_loss = 0

            for batch in self.loader:
                acc, loss = self.adversarial(task, batch[0], batch[1])
                accuracy += acc.item()
                total_loss += loss.item()

            accuracy /= len(self.loader)
            total_loss /= len(self.loader)

            self.accuracies.append(accuracy)
            self.losses.append(total_loss)
        else:
            self.accuracies.append(self.accumulator / self.count)
            self.losses.append(self.loss / self.count)
            self.distortions.append(self.distortion / self.count)
            self.count = 0
            self.distortion = 0
            self.accumulator = 0
            self.loss = 0

    def adversarial(self, task, batch, target):
        original_images = Variable(batch, requires_grad=True)

        # freeze model
        for param in task.model.parameters():
            param.requires_grad = False

        probabilities = task.model(original_images.to(device=task.device))
        loss = task.criterion(probabilities, target.to(device=task.device))
        loss.backward()

        pertubation = self.epsilon * torch.sign(original_images.grad)
        self.distortion += (pertubation.std() / original_images.std()).item()
        adversarial_images = batch + pertubation

        for param in task.model.parameters():
            param.requires_grad = True

        acc, loss = task.accuracy(adversarial_images, target)
        return acc, loss

    def on_new_batch(self, step, task, input, context):
        # make the examples
        batch, target = input

        acc, loss = self.adversarial(task, batch, target)

        self.loss += loss.item()
        self.accumulator += acc.item()
        self.count += 1

    def finish(self, task):
        if self.count > 0:
            self.on_new_epoch(None, task, None, None)

    def value(self):
        return {
            'adversary_accuracy': self.accuracies[-1],
            'adversary_loss': self.losses[-1],
            'adversary_distortion': self.distortions[-1]
        }
