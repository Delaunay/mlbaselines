from argparse import ArgumentParser

from olympus import fetch_device, StateStorage
from olympus.hpo import HPOptimizer, Fidelity
from olympus.datasets import Dataset, SplitDataset, DataLoader
from olympus.metrics import Accuracy
from olympus.models import Model
from olympus.optimizers import Optimizer, LRSchedule
from olympus.tasks import Classification
from olympus.tasks.hpo import HPO
from olympus.utils import option, show_dict


parser = ArgumentParser()
parser.add_argument('--epochs', type=int, default=3)
args = parser.parse_args()
device = fetch_device()
base = option('base_path', '/tmp/olympus')


def make_task():
    model = Model(
        'logreg',
        input_size=(1, 28, 28),
        output_size=(10,)
    )

    optimizer = Optimizer('sgd')

    lr_schedule = LRSchedule('exponential')

    data = Dataset('test-mnist', path=f'{base}/data')

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
        storage=StateStorage(folder=f'{base}/hpo_simple'))

    main_task.metrics.append(
        Accuracy(name='validation', loader=loader.valid(batch_size=64))
    )

    return main_task


space = make_task().get_space()

hp_optimizer = HPOptimizer(
    'hyperband',
    fidelity=Fidelity(1, 30).to_dict(),
    space=space)

hpo_task = HPO(hp_optimizer, make_task)

result = hpo_task.fit(objective='validation_accuracy')

print('Best Params:')
print('-' * 40)
print(f'validation_accuracy: {result.objective}')
show_dict(result.params)
