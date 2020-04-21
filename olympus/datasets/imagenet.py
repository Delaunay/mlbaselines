import functools
import os

import torch

from torchvision import transforms
from torchvision.datasets import ImageFolder

from olympus.datasets.dataset import AllDataset
from olympus.datasets.archive import ZipDataset


def default_transform():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


class ImagetNet(AllDataset):
    """TThe ImageNet project is a large visual database designed for use in visual object recognition software research.
    More than 14 million images have been hand-annotated by the project to indicate what objects are pictured and
    in at least one million of the images, bounding boxes are also provided.
    More on `wikipedia <https://en.wikipedia.org/wiki/ImageNet>`_.

    The full specification can be found at `here <http://www.image-net.org/>`_.

    Attributes
    ----------
    classes: List[int]
        Return the mapping between samples index and their class

    input_shape: (3, 224, 224)
        Size of a sample returned after transformation

    target_shape: (1000,)
        The classes are numbers from 0 to 999

    train_size: 14000000
        Size of the train dataset

    valid_size:
        Size of the validation dataset

    test_size:
        Size of the test dataset

    References
    ----------
    .. [1] Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
            Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei.(* = equal contribution)
            ImageNet Large Scale Visual Recognition Challenge

    """
    def __init__(self, data_path, image_folder=ImageFolder, train_size=None, valid_size=None, test_size=None,
                 input_shape=None, target_shape=None):
        transformations = default_transform()

        train_dataset = image_folder(os.path.join(data_path, 'train'))
        val_dataset = image_folder(os.path.join(data_path, 'val'))

        if test_size is None:
            test_size = len(val_dataset)

        super(ImagetNet, self).__init__(
            torch.utils.data.ConcatDataset([train_dataset, val_dataset]),
            test_size=test_size,
            train_size=train_size,
            valid_size=valid_size,
            transforms=transformations,
            input_shape=input_shape,
            target_shape=target_shape
        )

    @staticmethod
    def categories():
        return set(['classification'])   # 'detection'


def make_benzina_data_loader(args, size):
    import benzina.torch

    dataset = benzina.torch.ImageNet(args.data)

    return benzina.torch.DataLoader(
        dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        shape=(size, size),
        warp_transform=benzina.torch.operations.SimilarityTransform(),
        norm_transform=1 / 255,
        bias_transform=-0.5,
        shuffle=True
    )


builders = {
    'imagenet': ImagetNet,
    'imagenet_zip': functools.partial(ImagetNet, image_folder=ZipDataset),
}


def generate_jpeg_dataset(folder, shape=(3, 224, 224), num_class=1000, samples=192):
    """Generate a Fake JPEG Dataset for testing and benchmarking purposes"""
    from olympus.datasets.fake import FakeDataset
    import os

    fake = FakeDataset(shape, num_class, samples, 0, 0)
    os.makedirs(folder, exist_ok=True)

    for i, sample in enumerate(fake):
        target = sample['target']
        img = sample['batch']

        class_folder = f'{folder}/{target}'
        os.makedirs(class_folder, exist_ok=True)

        img.save(f'{class_folder}/{i}.jpg')
