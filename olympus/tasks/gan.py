
from dataclasses import dataclass

import torch

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer

from olympus.tasks.task import Task


@dataclass
class GAN(Task):
    generator: Module
    discriminator: Module
    generator_optimizer: Optimizer
    discriminator_optimizer: Optimizer
    latent_vector_size: int = 10
    criterion: Module = CrossEntropyLoss()

    def fit(self, step, input, context):
        self.generator.train()
        self.discriminator.train()

        batch, _ = input

        batch_size = batch.size(0)
        real_label = torch.full((batch_size,), 1, device=self.device)
        fake_label = torch.full((batch_size,), 0, device=self.device)

        # 1) Optimize the Discriminator
        #       maximize log(D(x)) + log(1 - D(G(z)))
        self.discriminator.zero_grad()

        # 1.a) Optimize with real Images
        prediction = self.discriminator(batch.to(device=self.device))
        discriminator_loss_real = self.criterion(prediction, real_label)
        discriminator_loss_real.backward(retain_graph=True)
        D_x = prediction.mean().detach()

        # 1.b) Optimizer with fake Images
        # Generate latent vector
        noise = torch.randn(batch_size, self.latent_vector_size, 1, 1, device=self.device)
        fake_images = self.generator(noise)

        prediction = self.discriminator(fake_images)
        discriminator_loss_fake = self.criterion(prediction, fake_label)
        discriminator_loss_fake.backward(retain_graph=True)
        D_G_z1 = prediction.mean().detach()

        discriminator_loss = discriminator_loss_fake + discriminator_loss_real
        self.discriminator_optimizer.step()

        # 2) Optimize the Generator
        #       maximize log(D(G(z)))
        self.generator.zero_grad()

        # fake_images = self.generator(noise)
        prediction = self.discriminator(fake_images)
        generator_loss = self.criterion(prediction, real_label)
        generator_loss.backward()

        D_G_z2 = prediction.mean().detach()

        self.generator_optimizer.step()

        return {
            'D_x': D_x,
            'D_G_z1': D_G_z1,
            'D_G_z2': D_G_z2,
            'G_loss': generator_loss.detach(),
            'D_loss': discriminator_loss.detach()
        }

    def generate(self, latent_vector):
        with torch.no_grad():
            self.generator.eval()
            return self.generator(latent_vector)

    def discriminate_probabilities(self, images):
        with torch.no_grad():
            self.discriminator.eval()
            return self.discriminator(images)

    def discriminate(self, batch):
        probabilities = self.discriminate_probabilities(batch)
        _, predicted = torch.max(probabilities, 1)
        return predicted

    def accuracy(self, batch, target):
        predicted = self.discriminate(batch)
        return (predicted == target).sum()

    def finish(self):
        pass
