import argparse
import itertools
from olympus.hpo import HPOptimizer, Fidelity, ParallelHPO


def train(uid, epoch, a, b, c, lr):
    return a + b + c + lr


def parallel_hpo(**kwargs):
    args = argparse.Namespace(**kwargs)

    # Arguments required for the HPO workers to synchronize
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int,
                        help='Worker rank, use to initialize the HPO')
    parser.add_argument('--uri', type=str, default='cockroach://192.168.0.1:8123',
                        help='Resource URI pointing to the database')
    parser.add_argument('--experiment', type=str, default='classification',
                        help='Database namespace to use for this experiment')

    parser.parse_args(namespace=args)

    params = {
        'a': 'uniform(0, 1)',
        'b': 'uniform(0, 1)',
        'c': 'uniform(0, 1)',
        'lr': 'uniform(0, 1)'
    }

    hpo = HPOptimizer('hyperband', fidelity=Fidelity(1, 30).to_dict(), space=params)

    # Wrap your HPO into Olympus ParallelHPO
    hpo = ParallelHPO(
        hpo,
        rank=args.rank,
        uri=args.uri,
        experiment=args.experiment)

    # Iterate over your configs distributed across workers
    for config in hpo:
        print('Worker: ', args.rank, config)
        validation_error = train(**config)
        hpo.observe(config, validation_error)

    # get the result of the HPO
    print(f'Worker {args.rank} is done')
    best_trial = hpo.result()
    if best_trial is not None:
        print(best_trial.params, best_trial.objective)


if __name__ == '__main__':
    # The code below is boilerplate to make the example work on your local machine without requiring
    # multiple nodes.

    # The script above assume you have a database running
    from olympus.hpo.worker import HPOWorkGroup

    # First time setup
    #   1. Launch a cockroach server
    #      Create the Namespace for our experiment
    uri = 'mongo://0.0.0.0:8123'
    namespace = 'trial'

    with HPOWorkGroup(uri, 'olympus', 'classification', clean=True, launch_server=True) as group:
        # Anywhere else; start as many clients as you want
        # we will use multiprocessing as an example but we can easily spawn clients on different nodes
        # as long as all the nodes can reach the database it will work
        from multiprocessing import Process

        count = 10
        workers = []
        for w in range(0, count):
            p = Process(
                target=parallel_hpo,
                kwargs=dict(uri=uri, namespace=namespace, rank=w)
            )
            p.start()
            workers.append(p)

        # wait for the client to finish
        for w in workers:
            w.join()
            print(f'closing {w}')
            w.close()

        # save the experiment for analysis
        group.archive('data.zip')
        group.stop()
