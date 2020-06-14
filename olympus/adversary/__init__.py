from olympus.utils import warning
from olympus.utils.factory import fetch_factories

from .gradient_ascent import GradientAscentAdversary
from .fast_gradient import FastGradientAdversary


registered_adversary = fetch_factories('olympus.adversary', __file__)


def known_adversary():
    return registered_adversary.keys()


def register_adversary(name, factory, override=False):
    global registered_adversary

    if name in registered_adversary:
        warning(f'{name} was already registered, use override=True to ignore')

        if not override:
            return

    registered_adversary[name] = factory


class RegisteredAdversaryNotFound(Exception):
    pass


class Adversary:
    def __init__(self, name, model, min_confidence=0.90, max_iter=10):
        ctor = registered_adversary.get(name)

        if ctor is None:
            raise RegisteredAdversaryNotFound(f'Adversary `{name}` does not exist')

        self.adversary = ctor(None, None, model, None)
        self.min_confidence = min_confidence
        self.max_iter = max_iter

    def __call__(self, batch):
        tempered_image, noise = self.adversary.generate(batch)

