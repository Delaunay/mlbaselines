from dataclasses import dataclass, field
import datetime
import json

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from olympus.utils.stat import StatStream


@dataclass
class Metric:
    """Metrics are observers that receives events periodically"""
    frequency_epoch: int = 0
    frequency_batch: int = 0

    def on_new_epoch(self, epoch, task, context):
        pass

    def on_new_batch(self, step, task, input, context):
        pass

    def start(self, task=None):
        pass

    def finish(self, task=None):
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
    """MetricList relays the Event to the Metrics/Observers"""
    def __init__(self, *args):
        self._metrics_mapping = dict()
        self.metrics = list()

        for arg in args:
            self.append(arg)

        self.batch_id: int = 0
        self._epoch: int = 0
        self._previous_step = 0

    def state_dict(self):
        return [m.state_dict() for m in self.metrics]

    def load_state_dict(self, state_dict):
        for m, m_state_dict in zip(self.metrics, state_dict):
            m.load_state_dict(m_state_dict)

    def __getitem__(self, item):
        v = self.get(item)

        if v is None:
            raise RuntimeError('Not found')

    def __setitem__(self, key, value):
        self.append(m=value, key=key)

    def get(self, key, default=None):
        if isinstance(key, int):
            return self.metrics[key]

        if isinstance(key, str):
            return self._metrics_mapping.get(key, default)

        return default

    def append(self, m: Metric, key=None):
        # Use name attribute as key
        if hasattr(m, 'name') and not key:
            key = m.name

        # Use type name as key
        elif not key:
            key = type(m).__name__

        # only insert if there are no conflicts
        if key not in self._metrics_mapping:
            self._metrics_mapping[key] = m

        self.metrics.append(m)

    def epoch(self, epoch, task=None, context=None):
        for m in self.metrics:
            if m.frequency_epoch > 0 and epoch % m.frequency_epoch == 0:
                m.on_new_epoch(epoch, task, context)

        self._epoch = epoch
        self.batch_id = 0

    def step(self, step, task=None, input=None, context=None):
        # Step back to 0, means it is a new epoch
        if self._previous_step > step:
            assert self.batch_id == 0

        for m in self.metrics:
            if m.frequency_batch > 0 and self.batch_id % m.frequency_batch == 0:
                m.on_new_batch(step, task, input, context)

        self.batch_id += 1
        self._previous_step = step

    def start(self, task=None):
        for m in self.metrics:
            m.start(task)

    def finish(self, task=None):
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


def get_time_delta(start):
    return (datetime.datetime.utcnow() - start).total_seconds()


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
        self.eval_time += (end - start).total_seconds()
        self.accuracies.append(acc / count)
        self.losses.append(loss_acc / count)

    def start(self, task=None):
        self.on_new_epoch(None, task, None)

    def finish(self, task=None):
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
            self.on_new_epoch(None, None, None)

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
            self.on_new_epoch(None, None, None)

    def value(self):
        return {
            self.name: self.metrics[-1]
        }


@dataclass
class ProgressView(Metric):
    print_fun = print
    epoch = 0
    step = 0
    max_epoch: int = 0
    max_step: int = 0
    batch_size: int = 0

    step_time = StatStream(drop_first_obs=5)
    epoch_time = StatStream(drop_first_obs=1)

    step_start = datetime.datetime.utcnow()
    epoch_start = datetime.datetime.utcnow()

    def show_progress(self, epoch, step=None):
        if step is None:
            step = '              '
        else:
            step = f'Step [{step:3d}/{self.max_step:3d}]'

        self.print_fun(f'\rEpoch [{epoch:3d}/{self.max_epoch:3d}] {step} {self.eta(epoch)}', end='')

    def eta(self, epoch):
        if self.step_time.count > 0:
            total_steps = self.max_step * self.max_epoch
            spent_steps = self.max_step * epoch + self.step
            remaining_steps = total_steps - spent_steps

            avg = self.step_time.avg
            # if we spent enough epochs estimate using both duration
            if self.epoch_time.count > 0:
                avg = (avg + self.epoch_time.avg / float(self.max_step)) / 2

            step_estimate = avg * remaining_steps
            return f'ETA: {step_estimate / 60:9.4f} min'

        return ''

    def on_new_epoch(self, epoch, task, context):
        self.epoch_time += get_time_delta(self.epoch_start)
        self.epoch_start = datetime.datetime.utcnow()

        self.epoch = epoch
        self.step = 0
        print()

        self.max_epoch = max(self.epoch, self.max_epoch)
        self.show_progress(epoch)
        print()

    def guess_batch_size(self, input):
        try:
            return input[0].shape[0]
        except Exception:
            return 0

    def on_new_batch(self, step, task, input, context):
        self.step_time += get_time_delta(self.step_start)
        self.step_start = datetime.datetime.utcnow()
        self.batch_size = self.guess_batch_size(input)

        self.step = step
        self.max_step = max(self.step, self.max_step)

        self.show_progress(self.epoch, step=self.step)

    def start(self, task=None):
        self.step_start = datetime.datetime.utcnow()
        self.epoch_start = self.step_start

    def finish(self, task=None):
        print()

    def value(self):
        result = {}

        if self.step_time.count > 0:
            result['step_time'] = self.step_time.avg
            if self.batch_size > 0:
                result['batch_speed'] = self.batch_size / self.step_time.avg

        if self.step_time.count > 2:
            result['step_time_sd'] = self.step_time.sd

        if self.epoch_time.count > 0:
            result['epoch_time'] = self.epoch_time.avg

        if self.epoch_time.count > 2:
            result['epoch_time_sd'] = self.epoch_time.sd

        return result

    def state_dict(self):
        return dict(
            max_epoch=self.max_epoch,
            max_step=self.max_step
        )

    def load_state_dict(self, state_dict):
        self.max_epoch = state_dict['max_epoch']
        self.max_step = state_dict['max_step']
        self.step = 0
        self.epoch = 0


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
    start_time: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    end_time: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    def start(self, task=None):
        pass

    def state_dict(self):
        return self.value()

    def load_state_dict(self, state_dict):
        self.start_time = self.end_time - datetime.timedelta(seconds=state_dict['elapsed_time'])

    def on_new_batch(self, step, task, input, context):
        self.end_time = datetime.datetime.utcnow()

    def finish(self, task=None):
        self.end_time = datetime.datetime.utcnow()

    @property
    def elapsed_time(self):
        return (self.end_time - self.start_time).total_seconds()

    def value(self):
        return {
            'elapsed_time': self.elapsed_time
        }


@dataclass
class ClassifierAdversary(Metric):
    """Simple Adversary Generator from `arxiv <https://arxiv.org/pdf/1412.6572.pdf.>`
    Measure how robust a network is from adversary attacks

    .. math::

        image = original_image + epsilon * sign(grad(cost(theta, original_image, t), original_image)

    Attributes
    ----------

    epislon: float = 0.25 (for mnist) 0.07 (for ImageNet)
        Epsilon corresponds to the magnitude of the smallest bit of an image encoding converted to real number

    References
    ----------
    .. [1] Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy.
        "Explaining and Harnessing Adversarial Examples", 20 Dec 2014

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
            self.on_new_epoch(None, task, None)

    def value(self):
        return {
            'adversary_accuracy': self.accuracies[-1],
            'adversary_loss': self.losses[-1],
            'adversary_distortion': self.distortions[-1]
        }
