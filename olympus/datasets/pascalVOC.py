from filelock import FileLock
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np

from olympus.datasets.dataset import AllDataset
from olympus.datasets.cache import DatasetCache
from olympus.utils import option


class ToLabel:
    def __call__(self, input):
        return torch.from_numpy(np.array(input, dtype=np.int64, copy=False))


class UserTransform:
    def __init__(self, input_sampling=None, target_sampling=None,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        self.target_transform = ToLabel()

    def __call__(self, input, target):
        input, target = self.input_transform(input), self.target_transform(target)
        return input, target


class DatasetPaddingWrapper(Dataset):
    def __init__(self, dataset, input_size=(512, 512), target_size=(512, 512)):
        super(DatasetPaddingWrapper, self).__init__()
        self.dataset = dataset
        self.input_size = input_size
        self.target_size = target_size

    def __getitem__(self, idx):
        # index into dataset
        input, target = self.dataset[idx]

        # calculate padding
        def get_padding(current_shape, new_shape):
            current_shape = torch.tensor(current_shape, dtype=torch.long)
            new_shape = torch.tensor(new_shape, dtype=torch.long)
            pad_total = new_shape - current_shape
            pad_low = pad_total // 2
            pad_high = pad_total - pad_low

            # format padding
            pad = torch.stack([pad_low, pad_high], dim=1)
            pad = torch.flip(pad, [0])
            pad = pad.view(-1)
            pad = tuple(pad)
            return pad

        input_padding = get_padding(input.size()[-2:], self.input_size)
        target_padding = get_padding(target.size()[-2:], self.target_size)

        # pad image / target
        input = F.pad(input, input_padding, value=0)
        target = F.pad(target, target_padding, value=255) # 255 is the 'void' label
        return input, target

    def __len__(self):
        return len(self.dataset)

class PascalVOC(AllDataset):
    """The PASCAL VOC segmentation dataset is a challenge dataset with the goal of segmenting 20 classes of objects
       in realistic scenes.

    The full specification can be found at `here <http://host.robots.ox.ac.uk/pascal/VOC/>`_.

    Using dataset provided in torchvision as documented here `here <https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.VOCSegmentation>`_.

    Attributes
    ----------
    input_shape: (3, 512, 512)
        Size of a sample returned after transformation

    target_shape: (512, 512)
        A segmentation map with class labels for each pixel

    train_size:
        Size of the train dataset

    valid_size:
        Size of the validation dataset

    test_size:
        Size of the test dataset

    References
    ----------
    .. [1] Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman, A.
           The PASCAL Visual Object Classes (VOC) Challenge.

    """
    def __init__(self, data_path, year='2012', cache=None, **kargs):

        with FileLock('voc.lock', timeout=option('download.lock.timeout', 4 * 60, type=int)):
            train_dataset = torchvision.datasets.VOCSegmentation(
                root=data_path, year=year, image_set='train', download=True,
                transforms=UserTransform())

        with FileLock('voc.lock', timeout=option('download.lock.timeout', 4 * 60, type=int)):
            test_dataset = torchvision.datasets.VOCSegmentation(
                root=data_path, year=year, image_set='val', download=True,
                transforms=UserTransform())

        dataset = DatasetPaddingWrapper(
            torch.utils.data.ConcatDataset([train_dataset, test_dataset]))

        if cache:
            dataset = DatasetCache(dataset, cache)

        if 'train_size' not in kargs:
            kargs['train_size'] = len(train_dataset)
        if 'valid_size' not in kargs:
            kargs['valid_size'] = len(test_dataset)//2
        if 'test_size' not in kargs:
            kargs['test_size'] = len(test_dataset)//2 + len(test_dataset)%2

        super(PascalVOC, self).__init__(
            dataset,
            **kargs,
        )

    @staticmethod
    def categories():
        return set(['segmentation'])   # 'detection'

    @staticmethod
    def nclasses():
        return 21

builders = {
    'voc-segmentation': PascalVOC,
}
