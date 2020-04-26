from dataclasses import dataclass, field
from typing import Callable, TypeVar

import torch
from torch import Tensor
import torch.nn.functional as F


from olympus.adversary.adversary import Adversary, AdversaryStat
from olympus.optimizers import Optimizer
from olympus.utils import debug, show_dict

Image = TypeVar('Image')


@dataclass
class GradientAscentAdversary(Adversary):
    """
    TODO: find original paper

    Parameters
    ----------
    model:
        Pytorch model

    target_class: int
        Index of the class we want the network to misclassify our image as

    preprocessor: Callable[[List[PILImage]], Tensor]
        use to apply preprocessing to images

    postprocesoor: Callable[[Tensor], List[PilImage]]
        used to reconstruct the image from a tensor

    Examples
    --------

    .. code-block:: python

        import torchvision.models as models
        from PIL import Image

        path = '/images/cat.jpg'
        img = Image.open(path)

        # img = torch.randn((1, 3, 224, 224))
        model = models.vgg19(pretrained=True)

        adversary = GradientAscentAdversary(
            imagenet_preprocess,
            imagenet_postprocessor,
            model,
            target_class=283)

        samples, noise = adversary.generate([img], min_confidence=0.90, lr=1)
        for s, n in zip(samples, noise):
            n.save('noise.jpg')
            s.save('adversary.jpg')

        show_dict(adversary.report())

    .. image:: ../../../docs/_static/images/cat.jpg
        :width: 30 %

    .. image:: ../../../docs/_static/images/gradient_ascent_noise.jpg
        :width: 30 %

    .. image:: ../../../docs/_static/images/gradient_ascent_adversary.jpg
        :width: 30 %

    """
    preprocessor: Callable[[Image], Tensor]
    postprocessor: Callable[[Tensor], Image]
    model: Callable[[Tensor], Tensor]
    target_class: int
    stats: AdversaryStat = field(default_factory=AdversaryStat)

    def confidence(self, image, target=None):
        probabilities = F.softmax(self.model(image), dim=1)
        idx = torch.argmax(probabilities)

        if target is None:
            return idx.item(), probabilities[0, idx].item()

        return idx.item(), probabilities[0, idx].item(), probabilities[0, target]

    def generate(self, image, min_confidence, lr=0.7, max_iter=500, round_trip=False):
        """Generate an adversarial example that should be misclassified as target_class

        Parameters
        ----------

        image: Union[Tensor, List[Image]]
            list of images to tamper with

        min_confidence: float
            Confidence we need to reach before stopping

        lr: float
            learning rate for the optimizer

        max_iter: int
            Maximal number of iteration

        round_trip: bool
            when enabled the tensor is periodically converted to image and back to tensor
        """
        self.model.eval()

        target = self.target_class

        if isinstance(image, list) and isinstance(image[0], Image.Image):
            original_image = self.preprocessor(image)
            input_image = original_image.clone()
        elif isinstance(image, torch.Tensor):
            original_image = image.clone()
            input_image = image
        else:
            raise RuntimeError('Expects Tensor or list of images')

        target_confidence = 0

        for i in range(max_iter):
            if target_confidence > min_confidence:
                break

            input_image.requires_grad = True

            optimizer = Optimizer(
                'sgd',
                params=[input_image],
                lr=lr,
                momentum=0,
                weight_decay=0)

            probabilities = F.softmax(self.model(input_image), dim=1)

            class_predicted = torch.argmax(probabilities)
            prediction_confidence = probabilities[0, class_predicted]
            target_confidence = probabilities[0, target]

            self.stats.update(
                class_predicted,
                prediction_confidence,
                target_confidence,
                probabilities)

            debug(f'{i:4d} Predicted {class_predicted} with {prediction_confidence:.4f},'
                  f'our target: {target} has {target_confidence.item():.4f}')

            self.model.zero_grad()
            optimizer.backward(-1 * target_confidence)
            optimizer.step()

            if round_trip:
                input_image = self.preprocessor(self.postprocessor(input_image))

        noises = self.get_noise(input_image, original_image)
        return self.postprocessor(input_image), noises


if __name__ == '__main__':
    import torchvision.models as models
    from PIL import Image
    from olympus.dashboard.plots.saliency import imagenet_preprocess, imagenet_postprocessor

    path = '/home/setepenre/work/olympus/docs/_static/images/cat.jpg'
    img = Image.open(path)

    # img = torch.randn((1, 3, 224, 224))
    model = models.vgg19(pretrained=True)

    adversary = GradientAscentAdversary(
        imagenet_preprocess,
        imagenet_postprocessor,
        model,
        target_class=283)

    samples, noise = adversary.generate([img], min_confidence=0.90, lr=1)
    for s, n in zip(samples, noise):
        n.save('noise.jpg')
        s.save('adversary.jpg')

    show_dict(adversary.report())
