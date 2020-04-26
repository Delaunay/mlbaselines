import torch
from torch.autograd import Variable
from torch import Tensor
import torch.nn.functional as F

from typing import Callable, TypeVar


Image = TypeVar('Image')


class GenerateAdversarialSample:
    preprocessor: Callable[[Image], Tensor]
    postprocessor: Callable[[Tensor], Image]
    model: Callable[[Tensor], Tensor]
    target_class: int
    criterion
    alpha: float


def generate(adversary, image, min_confidence, max_iter=500):
    adversary.model.eval()

    target = Variable(Tensor([adversary.target_class]))
    image = adversary.preprocessor(image)

    for _ in range(max_iter):
        torch.zero_gradients(image)

        out = adversary.model(image)
        loss = adversary.criterion(out, target)
        loss.backward()

        adversarial_noise = adversary.alpha * torch.sign(image.grad.detach())
        image += adversarial_noise

        # check if the network is failing
        image = adversary.preprocessor(adversary.postprocessor(image))
        out = adversary.model(image)

        _, prediction = torch.max(out, 1)
        confidence = F.softmax(out)[0][prediction].item()

        if prediction == adversary.target_class and confidence > min_confidence:
            break



