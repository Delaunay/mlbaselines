from sspace.space import compute_identity
from olympus.hpo.worker import HPOWorkGroup
from olympus.utils.testing import tiny_task


def main(study='lin_reg'):
    batch_sizes = [32, 64, 128, 256]
    lrs = [0.005, 0.1, 0.05, 0.01]
    seeds = [0, 1, 2, 3, 4, 5]
    epochs = [30]

    def arguments():
        for d in batch_sizes:
            for l in lrs:
                for e in epochs:
                    for s in seeds:
                        yield dict(epochs=e, batch_size=d, lr=l, seed=s)

    with HPOWorkGroup('mongo://127.0.0.1:27017', 'olympus', None) as group:
        group.launch_workers(2)
        group.clear_queue()
        group.client.monitor().clear('OLYMETRIC', group.experiment)

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
