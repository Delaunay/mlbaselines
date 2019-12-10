from argparse import ArgumentParser
import json

from olympus import fetch_device, StateStorage, TrackLogger
from olympus.datasets import Dataset, SplitDataset, DataLoader
from olympus.metrics import Accuracy
from olympus.models import Model
from olympus.optimizers import Optimizer, LRSchedule
from olympus.tasks import Classification
from olympus.tasks.hpo import HPO, fidelity


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

    data = Dataset('test-mnist', path='/tmp/olympus')

    splits = SplitDataset(data, split_method='original')

    loader = DataLoader(
        splits,
        sampler_seed=1,
        batch_size=32
    )

    main_task = Classification(
        classifier=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedule,
        dataloader=loader.train(),
        device=device,
        storage=StateStorage(folder='/tmp', time_buffer=0),
        logger=client)

    main_task.metrics.append(
        Accuracy(name='validation', loader=loader.valid(batch_size=64))
    )

    return main_task


client = TrackLogger(
    'hpo_simple',
    storage_uri='file://simple.json')

hpo = HPO(
    'minimalist_hpo',
    task=make_task,
    algo='ASHA',
    seed=1,
    num_rungs=2,
    num_brackets=1,
    max_trials=20,
    storage=f'track:file://simple.json'
)

hpo.fit(epochs=fidelity(args.epochs), objective='validation_accuracy')

print('Best Params:')
print('-' * 40)
print(json.dumps(hpo.best_trial.params, indent=2))
