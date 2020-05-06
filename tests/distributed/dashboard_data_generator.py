from sspace.space import compute_identity
from olympus.hpo.worker import HPOWorkGroup
from olympus.utils.testing import tiny_task


def main(study='lin_reg'):
    batch_sizes = [32, 64, 128, 256]
    lrs = [1, 0.5, 0.25, 0.01]
    epochs = [5]

    def arguments():
        for d in batch_sizes:
            for l in lrs:
                for e in epochs:
                    yield dict(epochs=e, batch_size=d, lr=l)

    with HPOWorkGroup('mongo://127.0.0.1:27017', 'olympus', None) as group:
        group.launch_workers(10)

        for i, kwargs in enumerate(arguments()):
            bs = kwargs['batch_size']
            lr = kwargs['lr']

            for j in range(10):
                kwargs['rs'] = j
                kwargs['uid'] = compute_identity(kwargs, 16)
                kwargs.pop('rs')
                group.queue_work(tiny_task, namespace=f'{study}-{bs}-{lr}', **kwargs)

        group.wait()


if __name__ == '__main__':
    main()
