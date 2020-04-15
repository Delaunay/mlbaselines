import time

from olympus.utils import debug, info
from olympus.utils.functional import select
from olympus.utils.importutil import get_import_path
from olympus.hpo.orion.adapter import HPOptimizer, hyperparameters, fidelity, Fidelity

from msgqueue.backends import new_server, new_client
from msgqueue.future import Future
from msgqueue.worker import BaseWorker
from msgqueue.worker import WORKER_JOIN, WORK_ITEM, WORK_QUEUE, \
    RESULT_ITEM, RESULT_QUEUE, WORKER_LEFT, SHUTDOWN


HPO_ITEM = 100


class HPOHelper:
    def __init__(self, module, function, fidelity=None, space=None, hpo_state=None, worker=0,  args=None, kwargs=None):
        hpo_state = select(hpo_state, {})

        if len(hpo_state) == 0:
            assert space is not None
            assert fidelity is not None

            hpo_state['space'] = space
            hpo_state['fidelity'] = Fidelity.from_dict(fidelity)

        self.hpo = HPOptimizer(**hpo_state)
        self.module = module
        self.function = function
        self.worker_count = worker
        self.kwargs = select(kwargs, {})
        self.args = select(args, [])
        self.sleep = 1
        self.shutdown_callback = None

    def state_dict(self):
        return {
            'worker': self.worker_count,
            'module': self.module,
            'function': self.function,
            'kwargs': self.kwargs,
            'args': self.args,
            'hpo_state': self.hpo.state_dict()
        }

    def fetch_results(self, client, m=None):
        new_results = []

        if m is None:
            m = client.pop(RESULT_QUEUE)

        while m is not None:
            if m.mtype == RESULT_ITEM:
                new_results.append(m.message)

            elif m.mtype == WORKER_JOIN:
                self.worker_count += 1

            elif m.mtype == WORKER_LEFT:
                self.worker_count -= 1

            else:
                debug(f'Received: {m}')

            client.mark_actioned(RESULT_QUEUE, m)
            m = client.pop(RESULT_QUEUE)

        debug(f'received {len(new_results)} new results')
        return new_results

    def insert_work(self, client, suggestions):
        for uid, params in suggestions:
            debug(f'sending {uid}')

            client.push(WORK_QUEUE, {
                'uid': uid,
                'args': [],
                'kwargs': params,
                'module': self.module,
                'function': self.function,
            }, mtype=WORK_ITEM)

    def insert_hpo(self, client):
        client.push(WORK_QUEUE, self.state_dict(), mtype=HPO_ITEM)

    def shutdown(self, client, callback=None):
        debug('sending shutdown signals')
        for _ in range(self.worker_count):
            client.push(WORK_QUEUE, {}, mtype=SHUTDOWN)

        if self.shutdown_callback:
            self.shutdown_callback()

    def kill_idle_workers(self, client):
        remaining = self.hpo.trial_count_remaining
        worker = self.worker_count

        # Keep a spare worker
        kill_worker = max(worker - (remaining + 1), 0)

        info(f'killing {kill_worker} workers because (worker: {worker}) > (remaining: {remaining}) ')

        for i in range(kill_worker):
            client.push(WORK_QUEUE, {}, mtype=SHUTDOWN)

    def step(self, client, message=None):
        # --- Fetch Results
        new_results = self.fetch_results(client, message)

        if len(new_results) == 0:
            debug('no new results')

        # --- Observe
        self.hpo.observe(new_results)

        # --- Suggest or Promote
        suggestions = self.hpo.suggest()

        # --- Should we shutdown
        if len(suggestions) == 0:
            debug('no new suggestions')
            if self.hpo.is_done:
                self.shutdown(client)
            else:
                time.sleep(self.sleep)

        # --- Insert Work
        self.insert_work(client, suggestions)

        # --- Kill idle workers
        if not self.hpo.is_done and len(suggestions) > 0:
            self.kill_idle_workers(client)


class TrialWorker(BaseWorker):
    def __init__(self, uri, namespace, id):
        super(TrialWorker, self).__init__(uri, namespace, id, WORK_QUEUE, RESULT_QUEUE)
        self.new_handler(WORK_ITEM, self.run_trial)
        self.new_handler(HPO_ITEM, self.run_hpo)
        self.new_handler(WORKER_JOIN, self.ignore_message)

    def run_trial(self, message, context):
        uid = message.message.get('uid', message.uid)

        args = message.message['args']
        kwargs = message.message['kwargs']

        module_name = message.message['module']
        function = message.message['function']

        module = __import__(module_name, fromlist=[''])
        fun = getattr(module, function)

        result = fun(*args, **dict(kwargs))
        debug(f'{self.client.name}: finished {uid} with {result}')
        return uid, kwargs, result

    def run_hpo(self, message, _):
        debug('new nomad hpo')
        hpo_state = message.message
        hpo = HPOHelper(**hpo_state)
        hpo.step(self.client)
        hpo.insert_hpo(self.client)
        debug('nomad hpo is killed')

    @staticmethod
    def sync_worker(*args, **kwargs):
        """Start the worker in the main process"""
        return TrialWorker(*args, **kwargs).run()

    @staticmethod
    def async_worker(*args, **kwargs):
        """Start a worker in a new process"""
        from multiprocessing import Process
        p = Process(target=TrialWorker.sync_worker, args=args, kwargs=kwargs)
        p.start()
        return p


class HPOWorker(BaseWorker):
    def __init__(self, uri, namespace, space, function):
        super(HPOWorker, self).__init__(uri, namespace, 'hpo', RESULT_QUEUE, WORK_QUEUE)
        self.new_handler(WORKER_JOIN, self.worker_join)
        self.new_handler(RESULT_ITEM, self.receive_results)
        self.new_handler(WORKER_LEFT, self.worker_left)

        self.helper = HPOHelper(
            space=space,
            module=function['module'],
            function=function['function'])

        suggestions = self.helper.hpo.suggest()
        self.helper.insert_work(self.client, suggestions)
        self.helper.sleep = 0.01
        self.helper.shutdown_callback = self.shutdown

    def shutdown(self):
        self.running = False

    def worker_join(self, message, context):
        self.helper.worker_count += 1

    def worker_left(self, message, context):
        self.helper.worker_count -= 1

    def receive_results(self, message, context):
        self.helper.step(self.client, message)

    @staticmethod
    def sync_hpo(*args, **kwargs):
        """Start the worker in the main process"""
        hpo = HPOWorker(*args, **kwargs)
        hpo.run()
        return hpo

    @staticmethod
    def async_hpo(*args, **kwargs):
        """Start a worker in a new process"""
        from multiprocessing import Process
        p = Process(target=HPOWorker.sync_hpo, args=args, kwargs=kwargs)
        p.start()
        return p


class HPOWorkGroup:
    def __init__(self, uri, namespace, worker_count, launch_server=False):
        # Start a message broker
        self.namespace = namespace
        self.uri = uri

        self.broker = None
        if launch_server:
            self.broker = new_server(uri)
            self.broker.start()

            self.broker.new_queue(namespace, WORK_QUEUE)
            self.broker.new_queue(namespace, RESULT_QUEUE)

        self.client = new_client(uri, namespace)
        self.client.name = 'group-leader'
        self.workers = []

        self.launch_workers(worker_count)

    def launch_workers(self, count):
        info('starting workers')
        for w in range(0, count):
            self.workers.append(TrialWorker.async_worker(self.uri, self.namespace, w))

    def launch_hpo(self, fun, *args, **kwargs):
        assert hasattr(fun, 'space'), 'Function need an hyper parameter space'
        info('starting HPO')

        function = {
            'module': get_import_path(fun.fun),
            'function': fun.fun.__name__
        }

        return HPOWorker.sync_hpo(self.uri, self.namespace, space=fun.space, function=function)

    def queue_hpo(self, space, fidelity, function, *args, **kwargs):
        module = get_import_path(function)
        function_name = function.__name__
        return self._queue_hpo(space, fidelity, module, function_name, *args, **kwargs)

    def _queue_hpo(self, space, fidelity, module, function, *args, **kwargs):
        return self.client.push(WORK_QUEUE, mtype=HPO_ITEM, message={
            'space': list(space.items()),
            'kwargs': kwargs,
            'args': args,
            'fidelity': fidelity.to_dict(),
            'module': module,
            'function': function
        })

    def optimize(self, fun, *args, **kwargs):
        assert hasattr(fun, 'space'), 'Function need an hyper parameter space'
        message_id = self.queue_hpo(fun.space, fun.fidelity, fun.fun, *args, **kwargs)
        return Future(self.client, WORK_QUEUE, message_id)

    def async_call(self, fun, *args, **kwargs):
        module = get_import_path(fun)
        function = fun.__name__

        msg_id = self.client.push(WORK_QUEUE, mtype=WORK_ITEM, message={
            'kwargs': kwargs,
            'args': args,
            'module': module,
            'function': function
        })

        return Future(self.client.cursor, RESULT_QUEUE, msg_id)

    def wait(self):
        for w in self.workers:
            w.join()
            w.close()
            info(f'joining worker{w}')

    def find(self, queue, target=-1):
        with self.client:
            hpo_state = self.client.pop(queue)
            self.client.mark_actioned(queue, hpo_state)

            while hpo_state is not None and hpo_state.mtype != target:
                hpo_state = self.client.pop(queue)
                self.client.mark_actioned(queue, hpo_state)

            return hpo_state

    def empty_queue(self, queue):
        return self.find(queue)

    def report(self):
        self.wait()

        # Get the last work item which is the HPO final state
        hpo_state = self.find(WORK_QUEUE, HPO_ITEM)
        self.empty_queue(RESULT_QUEUE)

        if self.broker is not None:
            self.broker.stop()

        if hpo_state is None:
            return None

        print('Trials:')
        hpo_state = hpo_state.message['hpo_state']
        trials = hpo_state['results']

        for uid, params, obj in trials:
            print(uid)
            print(f'  - {dict(params)}')
            print(f'  - {obj}')


@hyperparameters(epoch='fidelity(1, 30, 4)', lr='uniform(0, 1)', b='uniform(0, 1)', c='uniform(0, 1)')
def my_trial(epoch, lr, **kwargs):
    import time
    time.sleep(10 + epoch)
    return lr


def add(a, b):
    return a + b


def test_master_hpo(uri):
    """HPO is a special worker, what works along side the workers"""
    import logging
    from olympus.utils.log import set_log_level
    set_log_level(logging.DEBUG)

    group = HPOWorkGroup(
        uri, 'classification', worker_count=10)

    _ = group.launch_hpo(my_trial)

    group.report()

    return uri


def test_nomad_hpo(uri):
    """Worker are converted to HPO when new trials are needed then killed"""
    import logging
    from olympus.utils.log import set_log_level
    set_log_level(logging.DEBUG)

    group = HPOWorkGroup(
        uri, 'classification', worker_count=10, launch_server=False)

    group.optimize(my_trial)

    group.report()

    return uri


@hyperparameters(epoch='fidelity(1, 30, 4)', lr='uniform(0, 1)', b='uniform(0, 1)', c='uniform(0, 1)')
def dask_trial(uid, epoch, lr, **kwargs):
    import time
    time.sleep(10 + epoch)
    return uid, lr


def test_dask_poc():
    import logging
    from olympus.utils.log import set_log_level
    set_log_level(logging.DEBUG)

    from dask.distributed import Client
    from dask.distributed import as_completed

    client = Client()

    space = dask_trial.space
    hpo = HPOptimizer(
        space=space,
        fidelity=fidelity('epoch', 1, 30, 2),
        seed=1)

    while not hpo.is_done:
        fparams = {}
        futures = []

        for uid, params in hpo.suggest():
            info(f'sending trial {uid} {params}')
            future = client.submit(dask_trial.fun, uid, **dict(params))
            futures.append(future)
            fparams[uid] = params

        for batch in as_completed(futures, with_results=True).batches():
            for future, (uid, objective) in batch:
                info(f'received results {uid} with {objective}')
                hpo.observe(fparams[uid], objective)

    return


if __name__ == '__main__':
    import pymongo
    from msgqueue.uri import parse_uri

    uri = 'mongo://127.0.0.1:27017'
    # uri = 'cockroach://0.0.0.0:8123'

    opt = parse_uri(uri)
    c = pymongo.MongoClient(host=opt['address'], port=int(opt['port']))
    c.drop_database('classification')

    # test_hpo_optimizer_sample_different_hp()

    uri = test_nomad_hpo(uri)
    # uri = test_master_hpo(uri)

    # test_dask_poc()

    # broker = make_message_broker(uri)
    # broker.start()



