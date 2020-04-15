__descr__ = 'Hyper Optimized Model Zoo'
__version__ = '0.0.1'
__license__ = 'BSD-3-Clause'
__author__ = u'Pierre Delaunay'
__author_short__ = u'Delaunay'
__author_email__ = ('xavier.bouthillier@umontreal.ca', 'pierre@delaunay.io')
__copyright__ = u'2017-2019 Pierre Delaunay'
__url__ = 'https://github.com/mila-iqia/olympus'


from olympus.datasets import Dataset, SplitDataset, DataLoader
from olympus.metrics import Accuracy
from olympus.models import Model
from olympus.optimizers import Optimizer, LRSchedule

from olympus.tasks.hpo import HPO, fidelity
from olympus.tasks import Classification


from olympus.utils import fetch_device
from olympus.utils.storage import StateStorage
