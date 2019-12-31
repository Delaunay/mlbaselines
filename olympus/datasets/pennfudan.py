import os
import shutil

import numpy as np
import torch
from PIL import Image

from filelock import FileLock
from torchvision import datasets

from olympus.datasets.dataset import AllDataset
from olympus.utils.dtypes import VariableShape, Bound1D, DictionaryShape
from olympus.utils import option


class CocoDetection(datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


class _PennFudanDataset:
    """from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

    Notes
    -----
    BSD 3-Clause License

    Copyright (c) Soumith Chintala 2016,
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    URL = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'

    def __init__(self, data_path, transforms=None, target_transforms=None, download=True):
        self.root = data_path

        if download:
            with FileLock('penndufan.lock', timeout=option('download.lock.timeout', 4 * 60, type=int)):
                self.download()

        self.transforms = transforms
        self.target_transforms = target_transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(self.images_path)))
        self.masks = list(sorted(os.listdir(self.masks_path)))

    @property
    def images_path(self):
        return os.path.join(self.folder, 'PennFudanPed', 'PNGImages')

    @property
    def masks_path(self):
        return os.path.join(self.folder, 'PennFudanPed', 'PedMasks')

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.images_path, self.imgs[idx])
        mask_path = os.path.join(self.masks_path, self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        if self.target_transforms:
            target = self.target_transforms(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @property
    def folder(self):
        return os.path.join(self.root, self.__class__.__name__.replace('_', ''))

    def _check_exists(self):
        return os.path.exists(self.folder)

    def download(self):
        if self._check_exists():
            return

        try:
            from torchvision.datasets.utils import download_and_extract_archive

            os.makedirs(self.folder, exist_ok=True)
            url = _PennFudanDataset.URL
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.folder, filename=filename)

        except:
            shutil.rmtree(self.folder)
            raise


def penn_collate_fn(batch):
    return tuple(zip(*batch))


class PennFudanDataset(AllDataset):
    """This is an image database containing images that are used for pedestrian detection in the experiments reported in [1].
    The images are taken from scenes around campus and urban street.
    The objects we are interested in these images are pedestrians. Each image will have at least one pedestrian in it.

    All labeled pedestrians are straight up.
    More on `official website <https://www.cis.upenn.edu/~jshi/ped_html/>`_.

    Attributes
    ----------
    input_shape: (3, H, W) with H ∈ [311, 581], W ∈ [253, 1017] and H * W ∈ [81719, 451548]
        The heights of labeled pedestrians in this database fall into [180,390] pixels.
        They are all have unique shapes

    target_shape: DictionaryKeys('boxes', 'labels', 'masks', 'image_id', 'area', 'iscrowd')
        boxes: Tensor[P, 4] where P equals the number of pedestrian and 4 is the bounding box
        labels: Tensor[P], always 1
        masks: Tensor[P, H, W], 1 when the pixel belongs to a pedestrian, 0 if not
        image_id: Tensor[1], image id inside the dataset
        iscrowd: Tensor[P], if the image has a crowd of people (always false)

    train_size: 136
        Size of the train dataset
        96 images are taken from around University of Pennsylvania
        74 are taken from around Fudan University.

    valid_size: 16
        Size of the validation dataset

    test_size: 16
        Size of the test dataset

    References
    ----------
    .. [1] Liming Wang, Jianbo Shi, Gang Song, I-fan Shen.
        "Object Detection Combining Recognition and Segmentation". ACCV 2007
    """
    def __init__(self, data_path):
        from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip

        transforms = Compose([
            RandomHorizontalFlip(0.5),
            ToTensor()
        ])

        # FIXME: This is wrong validation and test should not have the horizontal flip
        super(PennFudanDataset, self).__init__(
            _PennFudanDataset(data_path, transforms),
            input_shape=VariableShape(C=3, H=Bound1D(311, 581), W=Bound1D(253, 1017)),
            target_shape=DictionaryShape('boxes', 'labels', 'masks', 'image_id', 'area', 'iscrowd'),
            train_size=136,
            test_size=16,
            valid_size=16
        )

    @property
    def num_classes(self):
        return 2

    @staticmethod
    def categories():
        return set(['detection'])

    collate_fn = penn_collate_fn


def _test_PennFudanDataset(*args, **kwargs):
    from torch.utils.data import Subset
    dataset = PennFudanDataset(*args, **kwargs)
    dataset.dataset = Subset(
        dataset.dataset,
        indices=list(range(0, 16))
    )
    dataset._train_size = 8
    dataset._test_size = 4
    dataset._valid_size = 4
    return dataset


builders = {
    'pennfudan': PennFudanDataset,
    'test_pennfudan': _test_PennFudanDataset,
}
