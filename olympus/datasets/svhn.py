import torch
from torchvision import datasets, transforms

from olympus.datasets.dataset import AllDataset


class SVHN(AllDataset):
    """SVHN is a real-world image dataset for developing machine learning and object recognition algorithms
    with minimal requirement on data preprocessing and formatting.
    It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits),
    but incorporates an order of magnitude more labeled data (over 600,000 digit images) and
    comes from a significantly harder, unsolved, real world problem (recognizing digits and
    numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images.
    More on `SVHN <http://ufldl.stanford.edu/housenumbers/>`_.

    See also :class:`.MNIST`

    Attributes
    ----------
    classes: List[int]
        Return the mapping between samples index and their class

    input_shape: (3, 32, 32)
        Size of a sample returned after transformation

    target_shape: (10,)
        The classes are numbers from 0 to 9

    train_size: 47225
        Size of the train dataset

    valid_size: 26032
        Size of the validation dataset

    test_size: 26032
        Size of the test dataset

    References
    ----------
    .. [1] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng.
        "Reading Digits in Natural Images with Unsupervised Feature Learning"
        NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011

    """
    def __init__(self, data_path):
        train_dataset = datasets.SVHN(
            data_path, split='train', download=True,
            transform=transforms.ToTensor())

        test_dataset = datasets.SVHN(
            data_path, split='test', download=True,
            transform=transforms.ToTensor())

        super(SVHN, self).__init__(
            torch.utils.data.ConcatDataset([train_dataset, test_dataset]),
            test_size=len(test_dataset)
        )


builders = {
    'svhn': SVHN}
