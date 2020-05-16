import copy
import time

from msgqueue.worker import WORKER_JOIN, WORKER_LEFT, SHUTDOWN, WORK_ITEM, RESULT_ITEM
from msgqueue.backends import new_client
from msgqueue.backends.queue import RecordQueue

from olympus.hpo.optimizer import OptimizationIsDone, WaitingForTrials, HyperParameterOptimizer
from olympus.hpo.utility import FunctionWithSpace
from olympus.utils.importutil import get_import_path
from olympus.utils import debug, info, option


WORK_QUEUE = 'OLYWORK'
RESULT_QUEUE = 'OLYRESULT'
HPO_ITEM = 100


def make_remote_call(function, *args, **kwargs):
    """Make a remote function call"""
    if isinstance(function, FunctionWithSpace):
        function = function.fun

    module = get_import_path(function)
    function_name = function.__name__
    return {
        'module': module,
        'function': function_name,
        'args': args,
        'kwargs': kwargs
    }


def exec_remote_call(state):
    """Execute a remote call"""
    args = state.get('args', [])
    kwargs = state.get('kwargs', {})

    module_name = state['module']
    function = state['function']

    module = __import__(module_name, fromlist=[''])
    fun = getattr(module, function)

    return fun(*args, **dict(kwargs))


class HPOManager:
    """Parallel HPO internals"""
    def __init__(self, client, state, backoff=0):
        self.client = client
        self.future_client = RecordQueue()
        self.state = state
        self.work = state['work']
        self.worker_count = state.get('worker_count', 0)
        self.experiment = state['experiment']
        self.backoff = backoff

    def shutdown(self):
        debug('sending shutdown signals')
        for _ in range(self.worker_count):
            self.future_client.push(WORK_QUEUE, self.experiment, {}, mtype=SHUTDOWN)

    def kill_idle_worker(self, hpo):
        remaining = hpo.remaining()
        worker = self.worker_count

        # Keep a spare worker
        kill_worker = max(worker - (remaining + 1), 0)
        info(f'killing {kill_worker} workers because (worker: {worker}) > (remaining: {remaining}) ')

        for i in range(kill_worker):
            self.future_client.push(WORK_QUEUE, self.experiment, {}, mtype=SHUTDOWN)

    def step(self, hpo):
        new_results = self.observe(hpo)
        new_trials = self.suggest(hpo)

        if hpo.is_done():
            self.shutdown()
            # Queue the HPO but this time in the result queue
            self.queue_hpo(hpo, RESULT_QUEUE)
            return 0, 0
        else:
            self.kill_idle_worker(hpo)

        if new_trials == 0:
            info(f'HPO sleeping {2 ** self.backoff} seconds')
            time.sleep(2 ** self.backoff)

        if 'hpo_state' in self.state:
            self.queue_hpo(hpo)

        return new_results, new_trials

    def pop_result(self):
        return self.client.pop(RESULT_QUEUE, self.experiment, mtype=(RESULT_ITEM, WORKER_JOIN, WORKER_LEFT))

    def observe(self, hpo):
        debug('observe')
        new_results = 0

        m = self.pop_result()
        while m is not None:
            if m.mtype == RESULT_ITEM:
                info(f'HPO {self.experiment} observed {m.message[0]["uid"]}')
                hpo.observe(m.message[0], m.message[1])
                new_results += 1

            elif m.mtype == WORKER_JOIN:
                self.worker_count += 1

            elif m.mtype == WORKER_LEFT:
                self.worker_count -= 1

            else:
                debug(f'Received: {m}')

            self.future_client.mark_actioned(RESULT_QUEUE, m)
            m = self.pop_result()
        return new_results

    def suggest(self, hpo):
        debug('suggest')
        trials = self._maybe_suggest(hpo)

        if trials is None:
            return 0

        for trial in trials:
            new_work = copy.deepcopy(self.work)
            new_work['kwargs'] = trial
            info(f'HPO {self.experiment} suggested {trial["uid"]}')
            self.future_client.push(WORK_QUEUE, self.experiment, new_work, mtype=WORK_ITEM)

        return len(trials)

    def recorded_operations(self):
        """Return all the operations that need to be performed for this task to be completed"""
        return self.future_client.records()

    @staticmethod
    def _maybe_suggest(hpo):
        try:
            return hpo.suggest()
        except OptimizationIsDone:
            return None
        except WaitingForTrials:
            return None

    def hpo_work_item(self, hpo):
        self.state['hpo_state'] = hpo.state_dict()
        self.state['worker_count'] = self.worker_count
        return self.state

    def queue_hpo(self, hpo, queue=WORK_QUEUE):
        self.future_client.push(queue, self.experiment, self.hpo_work_item(hpo), mtype=HPO_ITEM)

    def run(self, hpo):
        while not hpo.is_done():
            self.step(hpo)

    @staticmethod
    def sync_hpo(hpo, *args, **kwargs):
        """Start the worker in the main process"""
        manager = HPOManager(*args, **kwargs)
        manager.run(hpo)
        return manager

    @staticmethod
    def async_hpo(*args, **kwargs):
        """Start a worker in a new process"""
        from multiprocessing import Process
        p = Process(target=HPOManager.sync_hpo, args=args, kwargs=kwargs)
        p.start()
        return p


class ExperimentFinished(Exception):
    pass


class ParallelHPO:
    """Wraps an hyperparameter optimizer and make it work in collaboration with different HPO instances

    Parameters
    ----------
    hpo: HyperParameterOptimizer
        Optimizer instance that will be parallelized

    rank :int
        Worker ID, first worker will initialize the system

    uri: str
        Resource/Database used to synchronize workers

    experiment: str:
        unique string that identify this HPO

    database: str
        Name of the database the work queue will be created
    """

    def __init__(self, hpo, rank, uri, experiment, database=option('olympus.database', 'olympus')):
        self.hpo = hpo
        self.experiment = experiment
        self.client = new_client(uri, database)
        self.current_message = None

        # check that HPO is not finished
        state = self._fetch_final_state()
        if state is not None:
            raise ExperimentFinished(f'Experiment `{experiment}` is finished, change the experiment name')

        # first worker queue HPO
        if rank == 0:
            self._queue_hpo()

        # broadcast that one worker is joining
        self.client.push(RESULT_QUEUE, self.experiment, {}, mtype=WORKER_JOIN)

    def _queue_hpo(self):
        hpo = {
            'hpo': make_remote_call(type(self.hpo), **self.hpo.kwargs),
            'hpo_state': self.hpo.state_dict(),
            'worker_count': 0,
            'work': {},
            'experiment': self.experiment
        }
        return self.client.push(WORK_QUEUE, self.experiment, mtype=HPO_ITEM, message=hpo)

    def suggest(self, depth=0):
        """Pop an item from the work queue"""
        if depth > 0:
            time.sleep(1)

        # if depth > 10:
        #     raise WaitingForTrials(f'Retried to find new trials {depth} times without success')

        m = None
        while m is None:
            m = self.client.pop(WORK_QUEUE, self.experiment)

            if m is None:
                time.sleep(0.001)

        if m.mtype == HPO_ITEM:
            self.run_hpo(m)
            return self.suggest(depth + 1)

        elif m.mtype == WORK_ITEM:
            self.current_message = m
            return [m.message['kwargs']]

        elif m.mtype == SHUTDOWN:
            self.client.push(RESULT_QUEUE, self.experiment, {}, mtype=WORKER_LEFT)
            raise OptimizationIsDone()

        info(f'Received unsupported message {m}')
        return self.suggest(depth + 1)

    def observe(self, config, result):
        """Mark the previous message as actioned and push result"""
        assert self.current_message.message['kwargs'] == config
        self.client.push(RESULT_QUEUE, self.experiment, (config, result), mtype=RESULT_ITEM)
        self.client.mark_actioned(WORK_QUEUE, self.current_message)

    def run_hpo(self, message):
        state = message.message

        # Instantiate HPO
        self.hpo = exec_remote_call(state['hpo'])
        hpo_state = state.get('hpo_state')

        if hpo_state is not None:
            self.hpo.load_state_dict(hpo_state)

        manager = HPOManager(self.client, state)
        new_results, new_trials = manager.step(self.hpo)
        info(f'HPO read (results: {new_results}) and queued (trials: {new_trials})')

    def __getattr__(self, item):
        if hasattr(self.hpo, item):
            return getattr(self.hpo, item)

        raise AttributeError(f'{item} is not a known attribute of {self}')

    def __iter__(self):
        return HyperParameterOptimizer._ConfigurationIterator(self)

    def _fetch_final_state(self):
        # Wait for the HPO to be a result
        messages = self.client.monitor().unread_messages(RESULT_QUEUE, self.experiment)

        state = None
        for m in messages:
            if m.mtype == HPO_ITEM:
                state = m

        return state

    def result(self):
        state = self._fetch_final_state()

        if state is None:
            info('No HPO_ITEM message found')
            return None

        state = state.message
        self.hpo.load_state_dict(state['hpo_state'])
        return self.hpo.result()
