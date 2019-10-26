import time
import hashlib
import multiprocessing

from olympus.utils import error, info
from olympus.distributed import make_message_broker, make_message_client
from olympus.distributed.queue import MessageQueue

from orion.algo.base import BaseAlgorithm, OptimizationAlgorithm
from orion.core.io.space_builder import SpaceBuilder

#   {
#       'script': './executable.sh'
#       'args': ['--arg1', 'arg1']
#       'env': {'MY_ENV': 'VALUE'}
#   }
WORK_ITEM = 0

#   {
#       'kwargs': {
#           'uri': 'cockroach://192.168.0.10:8123'
#           ...
#       }
#   }
START_BROKER = 5
SHUTDOWN = 1
RESULT_MESSAGE = 2
WORKER_JOIN = 3
WORKER_LEFT = 4
START_HPO = 6

WORK_QUEUE = 'orion_work'
RESULT_QUEUE = 'orion_result'


default_asha = dict(
    seed=None,
    max_resources=100,
    grace_period=2,
    reduction_factor=4,
    num_brackets=1
)


class Sampler:
    def __init__(self, optimizer_name='ASHA', **kwargs):
        self.optimizer_args = kwargs
        if not kwargs:
            self.optimizer_args = default_asha

        self.optimizer_name = optimizer_name
        self.optimizer: BaseAlgorithm = None
        self.hpo_space = None
        self.samples = {}

    def hyperparameters(self, **kwargs):
        builder = SpaceBuilder()
        self.hpo_space = builder.build(kwargs)
        self.optimizer = OptimizationAlgorithm(self.optimizer_name, self.hpo_space, **self.optimizer_args)

    def suggest(self):
        suggestions = self.optimizer.suggest()

        results = []
        for suggestion in suggestions:
            args = {}
            uid = hashlib.sha256()

            for name, value in zip(self.hpo_space, suggestion):
                args[name] = value
                uid.update(str((name, value)).encode('utf8'))

            uid = uid.hexdigest()
            self.samples[uid] = args
            results.append((uid, args))

        return results[0]

    def observe(self, p, r):
        return True
        # return self.optimizer.observe(p, r)


class WorkScheduler:
    def __init__(self, message_queue, sampler, hyperparameters):
        self.trial_count = 0
        self.worker_count = 0
        self.max_trials = 10
        self.message_queue = message_queue
        self.client: MessageQueue = None
        self.sampler = Sampler(**sampler)
        self.sampler.hyperparameters(**hyperparameters)

    def sample(self):
        return self.sampler.suggest()

    def is_done(self):
        return self.client.actioned_count(WORK_QUEUE) > 10

    def insert_workitem(self):
        info('insert new work item')
        self.trial_count += 1
        self.client.push(WORK_QUEUE, message={
            'script': '/Tmp/work.sh',
            'args': self.sample(),
            'env': {}
        }, mtype=WORK_ITEM)

    def process_result(self):
        results = []
        result = self.client.pop(RESULT_QUEUE)

        self._observe(results, result)
        while result is not None:
            result = self.client.pop(RESULT_QUEUE)
            self._observe(results, result)

        return results

    def _observe(self, results, r):
        if r is None:
            return results

        elif r.mtype == RESULT_MESSAGE:
            info('found result')
            results.append(r)

        elif r.mtype == WORKER_JOIN:
            info('new worker')
            self.worker_count += 1

        elif r.mtype == WORKER_LEFT:
            info('worker died')
            self.worker_count -= 1

        else:
            error(f'Message not recognized (message: {r.uid})')

        info(f'actioned (message: {r.uid})')
        self.client.mark_actioned(RESULT_QUEUE, r)

    def run(self):
        info('starting scheduler')
        self.client = make_message_client(self.message_queue)
        self.client.name = 'HPO-scheduler'

        with self.client:
            while not self.is_done():
                max_trials_not_reached = self.trial_count <= self.max_trials
                worker_idle = self.client.unactioned_count(WORK_QUEUE) < self.worker_count
                insert_item = max_trials_not_reached and worker_idle

                if insert_item:
                    self.insert_workitem()

                results = self.process_result()

                if not results and not insert_item:
                    time.sleep(0.01)

            # shutdown whole system
            for _ in range(self.worker_count):
                self.client.push(WORK_QUEUE, message={}, mtype=SHUTDOWN)

            while self.client.unactioned_count(WORK_QUEUE) > 0:
                time.sleep(0.01)
