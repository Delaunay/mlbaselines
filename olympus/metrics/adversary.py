from dataclasses import dataclass, field
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from olympus.observers.observer import Metric
from olympus.utils.stat import StatStream


@dataclass
class ClassifierAdversary(Metric):
    """Simple Adversary Generator from `arxiv <https://arxiv.org/pdf/1412.6572.pdf.>`
    Measure how robust a network is from adversary attacks

    .. math::

        adversary(image) = image + epsilon * sign(grad(cost(theta, image, t), image)

    An adversary takes as input an image and returns a modified image that
    will try to induce an error on the classifier.

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
    time = StatStream(drop_first_obs=1)

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

        start = datetime.utcnow()
        acc, loss = self.adversarial(task, batch, target)
        self.time += (datetime.utcnow() - start).total_seconds

        self.loss += loss.item()
        self.accumulator += acc.item()
        self.count += 1

    def finish(self, task=None):
        if self.count > 0:
            self.on_new_epoch(None, task, None)

    def value(self):
        results = {
            'adversary_accuracy': self.accuracies[-1],
            'adversary_loss': self.losses[-1],
            'adversary_distortion': self.distortions[-1]
        }

        if self.time.count > 0:
            results['adversary_time'] = self.time.avg

        return results

    def state_dict(self):
        return dict(
            accuracies=self.accuracies,
            losses=self.losses,
            distortions=self.distortions
        )

    def load_state_dict(self, state_dict):
        self.accuracies = state_dict['accuracies']
        self.losses = state_dict['losses']
        self.losses = state_dict['distortions']

