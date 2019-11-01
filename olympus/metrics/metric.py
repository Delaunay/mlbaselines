from dataclasses import dataclass, field
import datetime
import json

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from olympus.utils.stat import StatStream


@dataclass
class Metric:
    frequency_epoch: int = 0
    frequency_batch: int = 0

    def on_new_epoch(self, epoch, task, context):
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
        self._epoch: int = 0
        self._previous_step = 0

    def state_dict(self):
        return [m.state_dict() for m in self.metrics]

    def load_state_dict(self, state_dict):
        for m, m_state_dict in zip(self.metrics, state_dict):
            m.load_state_dict(m_state_dict)

    def append(self, m: Metric):
        self.metrics.append(m)

    def epoch(self, epoch, task, context):
        for m in self.metrics:
            if m.frequency_epoch > 0 and epoch % m.frequency_epoch == 0:
                m.on_new_epoch(epoch, task, context)

        self._epoch = epoch
        self.batch_id = 0

    def step(self, step, task, input, context):
        # Step back to 0, means it is a new epoch
        if self._previous_step > step:
            assert self.batch_id == 0

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


@dataclass
class Accuracy(Metric):
    loader: DataLoader = None
    accuracies: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    frequency_epoch: int = 1
    frequency_batch: int = 0
    name: str = 'validation'
    eval_time: StatStream = field(default_factory=lambda: StatStream(drop_first_obs=0))
    total_time: int = 0

    def state_dict(self):
        return dict(accuracies=self.accuracies, losses=self.losses)

    def load_state_dict(self, state_dict):
        self.accuracies = state_dict['accuracies']
        self.losses = state_dict['losses']

    def on_new_epoch(self, epoch, task, context):
        acc = 0
        loss_acc = 0

        start = datetime.datetime.utcnow()

        count = len(self.loader)
        for data, target in self.loader:
            accuracy, loss = task.accuracy(data, target)

            acc += accuracy.item()
            loss_acc += loss.item()

        end = datetime.datetime.utcnow()
        self.eval_time += (end - start).seconds
        self.accuracies.append(acc / count)
        self.losses.append(loss_acc / count)

    def finish(self, task):
        self.on_new_epoch(None, task, None)

    def value(self):
        if not self.accuracies:
            return {}

        return {
            f'{self.name}_accuracy': self.accuracies[-1],
            f'{self.name}_loss': self.losses[-1],
            f'{self.name}_time': self.eval_time.avg
        }


@dataclass
class OnlineTrainAccuracy(Metric):
    """Reuse precomputed loss and prediction to get accuracy
    because the model is updated in between each batch, this does not return the true accuracy on the training set,
    """
    accuracies: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    accumulator: int = 0
    loss: int = 0
    count: int = 0

    frequency_epoch: int = 1
    frequency_batch: int = 1

    def state_dict(self):
        return dict(
            accuracies=self.accuracies,
            losses=self.losses,
            accumulator=self.accumulator,
            loss=self.loss,
            count=self.count
        )

    def load_state_dict(self, state_dict):
        self.accuracies = state_dict['accuracies']
        self.losses = state_dict['losses']
        self.accumulator = state_dict['accumulator']
        self.loss = state_dict['loss']
        self.count = state_dict['count']

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

    def on_new_epoch(self, epoch, task, context):
        if self.count > 0:
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
        if not self.accuracies:
            return {}

        return {
            'online_train_accuracy': self.accuracies[-1],
            'online_train_loss': self.losses[-1]
        }


@dataclass
class NamedMetric(Metric):
    name: str = None
    metrics: list = field(default_factory=list)
    accumulator = 0
    count = 0

    def on_new_epoch(self, epoch, task, context):
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

    def state_dict(self):
        return dict(epoch=self.epoch, sample_count=self.sample_count)

    def load_state_dict(self, state_dict):
        self.sample_count = state_dict['sample_count']
        self.epoch = state_dict['epoch']

    def on_new_epoch(self, epoch, task, context):
        self.epoch = epoch

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
    start: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    end: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    def state_dict(self):
        return self.value()

    def load_state_dict(self, state_dict):
        self.start = self.end - datetime.timedelta(seconds=state_dict['elapsed_time'])

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
    accuracies: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    distortions: list = field(default_factory=list)
    distortion = 0
    loss = 0
    accumulator = 0
    count = 0
    loader: DataLoader = None

    def on_new_epoch(self, epoch, task, context):
        if self.loader:
            accuracy = 0
            total_loss = 0

            for data, target in self.loader:
                acc, loss = self.adversarial(task, data, target)
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
