import zipfile
import time
from PIL import Image
from torch.utils.data.dataset import Dataset

from olympus.utils.stat import StatStream


def pil_loader(file_object):
    img = Image.open(file_object, 'r')
    img = img.convert('RGB')
    return img


class ZipDataset(Dataset):
    """Use a Zip archive to load a Folder dataset/ImageFolder.
    Zip are superiors to plain folders because they simplify access patterns for the OS

    Zip: 1 file with N random seeks
    Plain Folder: N files any where on the disk
    """

    def __init__(self, root, transform=None, target_transform=None,  loader=pil_loader):
        self.root = root
        self._zipfile = None
        self.loader = loader
        self.x_transform = transform
        self.y_transform = target_transform
        self._classes = None
        self._classes_to_idx = None
        self._files = None

        self._read_timer = StatStream(10)
        self._transform_timer = StatStream(10)

    def _init(self):
        a, b, c = self.find_classes(self.zipfile.namelist())
        self._classes = a
        self._classes_to_idx = b
        self._files = c

    @property
    def classes(self):
        if not self._classes:
            self._init()
        return self._classes

    @property
    def classes_to_idx(self):
        if not self._classes_to_idx:
            self._init()
        return self._classes_to_idx

    @property
    def files(self):
        if not self._files:
            self._init()
        return self._files

    @property
    def zipfile(self):
        """Zip file loading need to be delayed until the dataloader is ready,
         because the zip handles are not serializable
         """
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.root, 'r')

        return self._zipfile

    @property
    def read_timer(self):
        return self._read_timer

    @property
    def transform_timer(self):
        return self._transform_timer

    def find_classes(self, files):
        classes = set()
        classes_idx = {}
        nfiles = []

        for file in files:
            try:
                _, name, file_name = file.split('/')

                if file_name != '':
                    nfiles.append(file)

                if name not in classes:
                    classes_idx[name] = len(classes)
                    classes.add(name)

            except ValueError:
                pass

        return classes, classes_idx, nfiles

    def __getitem__(self, index):
        path = self.files[index]
        target = self.classes_to_idx[path.split('/')[1]]
        file = self.zipfile.open(self.files[index], 'r')

        s = time.time()
        sample = self.loader(file)
        self._read_timer += time.time() - s

        s = time.time()
        if self.x_transform is not None:
            sample = self.x_transform(sample)
        if self.y_transform is not None:
            target = self.y_transform(target)

        self._transform_timer += time.time() - s
        return sample, target

    def __len__(self):
        return len(self.files)
