from torch import Tensor
import torch.nn.functional as F

from olympus.optimizers import Optimizer

from typing import Callable, TypeVar


Image = TypeVar('Image')


class GenerateAdversarialSample:
    preprocessor: Callable[[Image], Tensor]
    postprocessor: Callable[[Tensor], Image]
    model: Callable[[Tensor], Tensor]
    target_class: int


def generate(adversary, image, min_confidence, lr, max_iter=500):
    adversary.model.eval()

    confidence = 0
    target = adversary.target_class

    for _ in range(max_iter):
        if confidence > min_confidence:
            break

        image = adversary.preprocessor(image)

        optimizer = Optimizer('sgd', params=[image], lr=lr)
        out = F.softmax(adversary.model(image))

        confidence = out[0][target].item()
        loss = -out[0, target]

        adversary.model.zero_grad()
        optimizer.step(loss)

        image = adversary.postprocessor(image)

