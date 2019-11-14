from argparse import ArgumentParser
import json

from olympus.datasets import DataLoader
from olympus.metrics import Accuracy
from olympus.models import Model
from olympus.optimizers import Optimizer
from olympus.optimizers.schedules import LRSchedule
from olympus.tasks.hpo import HPO, fidelity
from olympus.tasks import Classification
from olympus.utils.storage import StateStorage
from olympus.utils import fetch_device


parser = ArgumentParser()
parser.add_argument('--epochs', type=int, default=3)
args = parser.parse_args()
device = fetch_device()


def make_task():
    model = Model(
        'logreg',
        input_size=(1, 28, 28),
        output_size=(10,)
    )

    optimizer = Optimizer('sgd')

    lr_schedule = LRSchedule('exponential')

    dataset = DataLoader('test-mnist', {'name': 'original'})

    main_task = Classification(
        classifier=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedule,
        dataloader=dataset.train(),
        device=device,
        storage=StateStorage(folder='/tmp', time_buffer=0))

    main_task.metrics.append(
        Accuracy(name='validation', loader=dataset.valid())
    )

    return main_task


hpo = HPO(
    'minimalist_hpo',
    task=make_task,
    algo='ASHA',
    seed=1,
    num_rungs=3,
    num_brackets=1,
    max_trials=500
)

hpo.fit(epochs=fidelity(args.epochs), objective='validation_accuracy')

print('Best Params:')
print('-' * 40)
print(json.dumps(hpo.best_trial.params, indent=2))
