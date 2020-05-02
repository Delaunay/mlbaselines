from msgqueue.worker import BaseWorker, WORKER_JOIN, SHUTDOWN, WORK_ITEM
from msgqueue.backends import new_server, new_client

from olympus.hpo.parallel import WORK_QUEUE, RESULT_QUEUE, HPO_ITEM
from olympus.hpo.parallel import exec_remote_call, make_remote_call, HPOManager
from olympus.utils import info, option

# One of the issue with the explicit worker is that it requires:
# the hpo and the function to be import-able


class TrialWorker(BaseWorker):
    """Polyvalent worker that can do execute trials and HPO

    Parameters
    ----------
    uri: str
        Database address to connect to

    database: str
        Name of the database

    id: str
        Worker ID

    experiment: str
        Experiment this worker should be working on, if None will work on all experiments.
        If no experiments are specified then it will shutdown only in case of timeout

    hpo_allowed: bool
        Can HPO run on this worker

    work_allowed: bool
        Can work items run on this worker

    """
    def __init__(self, uri, database, id, experiment=None, hpo_allowed=True, work_allowed=True):
        super(TrialWorker, self).__init__(uri, database, experiment, id, WORK_QUEUE, RESULT_QUEUE)
        self.namespaced = experiment is not None

        if work_allowed:
            self.new_handler(WORK_ITEM, self.run_trial)

        if hpo_allowed:
            self.new_handler(HPO_ITEM, self.run_hpo)

        self.new_handler(WORKER_JOIN, self.ignore_message)

        self.timeout = option('worker.timeout', 5 * 60, type=int)
        self.max_retry = option('worker.max_retry', 3, type=int)

        # Disable shutting down when receiving shut down
        if experiment is None:
            info(f'Disabling message shutdown because {experiment}')
            self.dispatcher[SHUTDOWN] = lambda *args, **kwargs: print('ignoring shutdown signal')

    def run_trial(self, message, context):
        """Run a trial and return its result"""
        state = message.message
        uid = state['kwargs']['uid']
        state['kwargs']['experiment_name'] = context['namespace']
        state['kwargs']['client'] = self.client
        result = exec_remote_call(state)
        state['kwargs'].pop('experiment_name')
        state['kwargs'].pop('client')

        info(f'Finished (trial: {uid}) with (objective: {result:.5f})')
        return state['kwargs'], result

    def run_hpo(self, message, _):
        """Run the HPO only when needed and then let it die until the results are ready"""
        state = message.message

        # Instantiate HPO
        hpo = exec_remote_call(state['hpo'])
        hpo_state = state.get('hpo_state')

        if hpo_state is not None:
            hpo.load_state_dict(hpo_state)

        manager = HPOManager(self.client, state)
        new_results, new_trials = manager.step(hpo)
        info(f'HPO read (results: {new_results}) and queued (trials: {new_trials})')

    @staticmethod
    def sync_worker(uri, database, id, experiment):
        """Start the worker in the main process"""
        return TrialWorker(uri, database, id, experiment=experiment).run()

    @staticmethod
    def async_worker(uri, database, id, experiment):
        """Start a worker in a new process"""
        from multiprocessing import Process
        p = Process(target=TrialWorker.sync_worker, args=(uri, database, id, experiment))
        p.start()
        return p


class HPOWorkGroup:
    """Local parallel HPO helper"""

    def __init__(self, uri, database, experiment, clean=False, launch_server=False):
        # Start a message broker
        self.database = database
        self.uri = uri

        self.broker = None
        if launch_server:
            self.broker = new_server(uri, database)
            self.broker.start()

        self.client = new_client(uri, database)
        self.client.name = 'group-leader'
        self.experiment = experiment
        self.workers = []

        if clean:
            self.clear_queue()

        # self.launch_workers(worker_count)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def clear_queue(self):
        mn = self.client.monitor()
        mn.clear(WORK_QUEUE, self.experiment)
        mn.clear(RESULT_QUEUE, self.experiment)

    def wait(self):
        for w in self.workers:
            w.join()
            w.close()
            info(f'joining worker{w}')

    def stop(self):
        count = 0
        for w in self.workers:
            try:
                if w.is_alive():
                    count += 1
            except ValueError:
                # ValueError: process object is closed
                pass

        # kill all the workers
        # this should not be necessary because the HPO tries to kill workers
        # when it is done
        for _ in range(count):
            self.client.push(WORK_QUEUE, self.experiment, {}, mtype=SHUTDOWN)

        for w in self.workers:
            try:
                if w.is_alive():
                    w.terminate()
                    w.join()

                w.close()
            except ValueError:
                # ValueError: process object is closed
                pass

        if self.broker is not None:
            self.broker.stop()

    def launch_workers(self, count, namespaced=True):
        """Launching async workers"""
        info('starting workers')
        namespace = self.experiment
        if not namespaced:
            namespace = None

        for w in range(0, count):
            self.workers.append(TrialWorker.async_worker(self.uri, self.database, w, namespace))

    def queue_hpo(self, remote_hpo, remote_fun):
        """Queue the HPO as a worker"""
        hpo = {
            'hpo': remote_hpo,
            'hpo_state': None,
            'work': remote_fun,
            'experiment': self.experiment
        }
        return self.client.push(WORK_QUEUE, self.experiment, mtype=HPO_ITEM, message=hpo)

    def run_hpo(self, hpo, fun, *args, **kwargs):
        """Launch a master HPO"""
        state = {
            'work': make_remote_call(fun, *args, **kwargs),
            'experiment': self.experiment
        }
        manager = HPOManager.sync_hpo(hpo, self.client, state)
        return manager

    def archive(self, name, namespace_out=None, format='bson'):
        """Archive all the information generated during runtime and package it into a zip file"""
        self.client.monitor().archive(self.experiment, name, namespace_out=namespace_out, format=format)


def main():
    """HPO worker entry point"""
    import argparse
    import os
    import logging

    from olympus.utils.log import set_log_level
    set_log_level(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--uri', type=str, default='cockroach://0.0.0.0:8123',
                        help='Queue URI to use to dispatch work')

    parser.add_argument('--database', type=str, default='olympus',
                        help='name of the database')

    parser.add_argument('--experiment', type=str, default=None,
                        help='name of the experiment if we want the worker to be exclusive')

    parser.add_argument('--rank', type=int, default=os.getpid(),
                        help='Rank or ID of the worker, defaults to PID')

    parser.add_argument('--hpo-allowed', type=bool, default=True,
                        help='Can HPO run on this worker')

    parser.add_argument('--work-allowed', type=bool, default=True,
                        help='Can trials run on this worker')

    args = parser.parse_args()
    worker = TrialWorker(args.uri, args.database, args.rank, args.experiment)
    return worker.run()


if __name__ == '__main__':
    main()
