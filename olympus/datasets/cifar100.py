import torch
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image

from olympus.datasets.dataset import AllDataset


class CIFAR100(AllDataset):
    """See :class:`.CIFAR10`

    The full specification can be found at `here <https://www.cs.toronto.edu/~kriz/cifar.html>`_.

    ==============================  =====================================================
                Superclass 	                    Classes
    ==============================  =====================================================
    aquatic mammals 	            beaver, dolphin, otter, seal, whale
    fish 	                        aquarium fish, flatfish, ray, shark, trout
    flowers 	                    orchids, poppies, roses, sunflowers, tulips
    food containers 	            bottles, bowls, cans, cups, plates
    fruit and vegetables 	        apples, mushrooms, oranges, pears, sweet peppers
    household electrical devices 	clock, computer keyboard, lamp, telephone, television
    household furniture 	        bed, chair, couch, table, wardrobe
    insects     	                bee, beetle, butterfly, caterpillar, cockroach
    large carnivores 	            bear, leopard, lion, tiger, wolf
    large man-made outdoor things 	bridge, castle, house, road, skyscraper
    large natural outdoor scenes 	cloud, forest, mountain, plain, sea
    large omnivores and herbivores 	camel, cattle, chimpanzee, elephant, kangaroo
    medium-sized mammals 	        fox, porcupine, possum, raccoon, skunk
    non-insect invertebrates 	    crab, lobster, snail, spider, worm
    people 	                        baby, boy, girl, man, woman
    reptiles 	                    crocodile, dinosaur, lizard, snake, turtle
    small mammals 	                hamster, mouse, rabbit, shrew, squirrel
    trees 	                        maple, oak, palm, pine, willow
    vehicles 1          	        bicycle, bus, motorcycle, pickup truck, train
    vehicles 2 	                    lawn-mower, rocket, streetcar, tank, tractor
    ==============================  =====================================================

    Attributes
    ----------
    classes: List[int]
        Return the mapping between samples index and their class

    input_shape: (3, 32, 32)
        Size of a sample stored in this dataset

    target_shape: (100,)
        There are 100 classes see above for a full description

    train_size: 40000
        Size of the train dataset

    valid_size: 10000
        Size of the validation dataset

    test_size: 10000
        Size of the test dataset

    References
    ----------
    .. [1] Alex Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", 2009.

    """
    def __init__(self, data_path):
        transformations = [
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

        train_transform = [
            to_pil_image,
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()] + transformations

        transformations = dict(
            train=transforms.Compose(train_transform),
            valid=transforms.Compose(transformations),
            test=transforms.Compose(transformations))

        train_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transforms.ToTensor())

        super(CIFAR100, self).__init__(
            torch.utils.data.ConcatDataset([train_dataset, test_dataset]),
            test_size=len(test_dataset),
            transforms=transformations
        )

    @staticmethod
    def categories():
        return set(['classification'])


builders = {
    'cifar100': CIFAR100}
