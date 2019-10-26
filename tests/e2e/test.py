import hashlib
import multiprocessing
from orion.algo.base import BaseAlgorithm, OptimizationAlgorithm
from orion.core.io.space_builder import SpaceBuilder


default_asha = dict(
    seed=None,
    max_resources=100,
    grace_period=2,
    reduction_factor=4,
    num_brackets=1
)


class Orion:
    def __init__(self, optimizer_name='ASHA', **kwargs):
        self.optimizer_args = kwargs
        if not kwargs:
            self.optimizer_args = default_asha

        self.optimizer_name = optimizer_name
        self.optimizer: BaseAlgorithm = None
        self.hpo_space = None
        self.samples = {}

    def hyper_parameters(self, **kwargs):
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


if __name__ == '__main__':
    import time

    hpo = Orion('ASHA')

    hpo.hyperparameters(
        batch_size='choices(32, 40, 48, 56, 64)',
        lr='uniform(1e-5, 1)',
        epochs='fidelity()'
    )

    with multiprocessing.Manager() as manager:
        work_queue, result_queue = manager.Queue(), manager.Queue()

        def worker_task(*args, work_queue, result_queue, **kwargs):
            data = work_queue.get(timeout=1)

            if data is not None:
                uid, args = data
                time.sleep(5)

                a = result_queue.put({'uid': uid, 'result': 2})

        worker_pool = multiprocessing.Pool(processes=5)

        max_round = 2
        hpo_round = 0

        while hpo_round < max_round:

            # Schedule 10 work item
            for i in range(0, 10):
                work_queue.put(hpo.suggest())
                worker_pool.apply_async(
                    worker_task, kwds=dict(work_queue=work_queue, result_queue=result_queue))

            # wait for work item to be finished
            received_results = 0
            while received_results < 10:
                result = result_queue.get(timeout=10)

                if result is None:
                    raise TimeoutError()

                uid = result['uid']
                result = result['result']
                hpo.observe(uid, result)
                received_results += 1

            hpo_round += 1
