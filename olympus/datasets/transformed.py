import numbers
import random

from torch.utils.data.dataset import Subset
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.functional as F
from torchvision import transforms

from olympus.resuming import state_dict
from olympus.utils import compress_dict, decompress_dict


class TransformedSubset(Subset):
    def __init__(self, dataset, indices, transform=None):
        super(TransformedSubset, self).__init__(dataset, indices)

        self.transform = transform

    def __getitem__(self, idx):
        data = super(TransformedSubset, self).__getitem__(idx)
        target = data[-1]
        data = data[:-1]

        if self.transform is not None:
            data = [self.transform(x) for x in data]

        return data, target


# NOTE: Copied over from torchvision. Should consider contributing to it directly.
class RandomCrop(object):
    def __init__(self, size, seed=None, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.rng = random.Random(seed)

    @staticmethod
    def get_params(rng, img, output_size):
        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = rng.randint(0, h - th)
        j = rng.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(self.rng, img, self.size)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

    def load_state_dict(self, state):
        self.rng.setstate(state['rng'])

    def state_dict(self, compressed=True):
        return {'rng': self.rng.getstate()}


# NOTE: Copied over from torchvision. Should consider contributing to it directly.
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5, seed=None):
        self.p = p
        self.rng = random.Random(seed)

    def __call__(self, img):
        if self.rng.random() < self.p:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

    def load_state_dict(self, state):
        self.rng.setstate(state['rng'])

    def state_dict(self, compressed=True):
        return {'rng': self.rng.getstate()}


# NOTE: Copied over from torchvision. Should consider contributing to it directly.
def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class Compose(transforms.Compose):
    def load_state_dict(self, state):
        for transform, transform_state in zip(self.transforms, state['transforms']):
            if transform_state:
                transform.load_state_dict(transform_state)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        states = []
        for transform in self.transforms:
            if hasattr(transform, 'state_dict'):
                states.append(transform.state_dict())
            else:
                states.append(None)

        return {'transforms': tuple(states)}
