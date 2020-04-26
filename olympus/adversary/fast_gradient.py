from dataclasses import dataclass, field
from typing import Callable, TypeVar

import torch
from torch import Tensor
import torch.nn.functional as F

from olympus.adversary.adversary import Adversary, AdversaryStat
from olympus.utils import debug


Image = TypeVar('Image')


@dataclass
class FastGradientAdversary(Adversary):
    preprocessor: Callable[[Image], Tensor]
    postprocessor: Callable[[Tensor], Image]
    model: Callable[[Tensor], Tensor]
    target_class: int
    alpha: float
    stats: AdversaryStat = field(default_factory=AdversaryStat)

    def generate(self, image, min_confidence, max_iter=10, round_trip=False):
        from torch.autograd import Variable

        self.model.eval()

        target = torch.as_tensor([self.target_class])
        image = self.preprocessor(image)

        # Save the original to compute the final noise
        original_image = image

        for i in range(max_iter):
            image = Variable(image, requires_grad=True)
            image.grad = None

            # Compute the current confidence
            output = self.model(image)
            self.model.zero_grad()

            with torch.no_grad():
                probabilities = F.softmax(output, dim=1)
                prediction = torch.argmax(probabilities)
                confidence = probabilities[0, prediction]
                target_confidence = probabilities[0, self.target_class]

            self.stats.update(
                prediction,
                confidence,
                target_confidence,
                probabilities)

            debug(f'{i:4d} Predicted {prediction} with {confidence:.4f},'
                  f'our target: {self.target_class} has {target_confidence.item():.4f}')

            if prediction == self.target_class and confidence > min_confidence:
                break

            # Tampering some more
            loss = F.cross_entropy(output, target)
            loss.backward()

            adversarial_noise = self.alpha * torch.sign(image.grad.detach())
            image = image + adversarial_noise

            if round_trip:
                image = self.preprocessor(self.postprocessor(image))

        noises = self.get_noise(image, original_image)
        return self.postprocessor(image), noises


if __name__ == '__main__':
    from olympus.dashboard.plots.saliency import imagenet_preprocess, imagenet_postprocessor
    import torchvision.models as models
    from PIL import Image
    from olympus.utils import show_dict

    path = '/home/setepenre/work/olympus/docs/_static/images/cat.jpg'
    img = Image.open(path)

    # img = torch.randn((1, 3, 224, 224))
    model = models.vgg19(pretrained=True)

    adversary = FastGradientAdversary(
        imagenet_preprocess,
        imagenet_postprocessor,
        model,
        alpha=0.20,
        target_class=283)

    s, n = adversary.generate([img], min_confidence=0.90)
    n[0].save('noise.jpg')
    s[0].save('adversary.jpg')

    show_dict(adversary.report())
