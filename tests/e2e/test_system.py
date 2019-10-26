from olympus.prime import PrimeMonitor
from olympus.worker import Worker


def remove(name):
    try:
        shutil.rmtree(name)
    except FileNotFoundError:
        pass


if __name__ == '__main__':
    import shutil
    from multiprocessing import Process

    remove('/tmp/cockroach/queue')
    remove('/tmp/mongo')

    # Create the initial broker to initialize the system
    mq = 'mongodb://192.168.0.10:8123'
    init = PrimeMonitor(mq)

    # start a pool of workers
    def make_worker(id):
        w = Worker(mq, worker_id=id)
        w.run()
        return w

    workers = []
    for i in range(3):
        w = Process(target=make_worker, args=(i,))
        w.start()

    # init the pool of workers
    # Restore previous session if any
    init.restore_session()

    # queue two brokers for redundancy
    # in our case we are using a single node so we ignore that
    # master.queue_broker()
    # master.queue_broker()

    # queue HPO creation
    init.queue_hpo()

    # wait for HPO to finish
    init.wait()

    # shutdown system
    init.shutdown()

