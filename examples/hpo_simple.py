from olympus.tasks.hpo import HPO
from olympus.tasks import Classification
from olympus.optimizers import Optimizer
from olympus.models import Model
from olympus.optimizers.schedules import LRSchedule
from olympus.datasets import DataLoader
from olympus.utils.storage import StateStorage
from olympus.utils import fetch_device
from olympus.metrics import Accuracy

device = fetch_device()


def make_task():
    model = Model(
        'resnet18',
        input_size=(1, 28, 28),
        output_size=(10,)
    ).to(device=device)

    optimizer = Optimizer('sgd')

    lr_schedule = LRSchedule('exponential')

    dataset = DataLoader('mnist', {'name': 'original'})

    main_task = Classification(
        classifier=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedule,
        dataloader=dataset.train(),
        device=device,
        storage=StateStorage(folder='/tmp'))

    main_task.metrics.append(
        Accuracy(name='validation', loader=dataset.valid())
    )

    return main_task


hpo = HPO(make_task)
hpo.run()
