import subprocess
import traceback
import time

from olympus.utils import error, info
from olympus.distributed import make_message_broker, make_message_client
from olympus.distributed.queue import MessageQueue, Message
from olympus.hpo import WORK_QUEUE, RESULT_QUEUE
from olympus.hpo import WORKER_LEFT, WORKER_JOIN
from olympus.hpo import START_BROKER, START_HPO, SHUTDOWN, WORK_ITEM
from olympus.hpo import WorkScheduler


class Worker:
    def __init__(self, queue_uri, worker_id):
        self.uri = queue_uri
        self.client: MessageQueue = make_message_client(queue_uri)
        self.running = False
        self.work_id = worker_id
        self.broker = None

    def run(self):
        info('starting worker')

        self.running = True
        self.client.name = f'worker-{self.work_id}'
        self.client.push(RESULT_QUEUE, message={'worker': '1'}, mtype=WORKER_JOIN)
        last_message = None

        with self.client:
            while self.running:
                try:
                    workitem = self.client.pop(WORK_QUEUE)

                    if workitem is None:
                        time.sleep(0.01)

                    elif workitem.mtype == START_BROKER:
                        info('starting new message broker')
                        msg = workitem.message
                        # self.broker = make_message_broker(**msg.get('kwargs'))
                        # self.broker.start()

                    elif workitem.mtype == WORK_ITEM:
                        self.execute(workitem)

                    elif workitem.mtype == SHUTDOWN:
                        info(f'shutting down worker')
                        self.running = False
                        last_message = workitem

                    # Shutdown worker loop and start HPO that has it's own loop
                    elif workitem.mtype == START_HPO:
                        info('starting HPO service')
                        self.running = False
                        last_message = workitem

                    else:
                        error(f'Unrecognized (message: {workitem})')

                except Exception:
                    error(traceback.format_exc())
        # --

        self.client.push(RESULT_QUEUE, message={'worker': '0'}, mtype=WORKER_LEFT)
        if last_message:
            self.client.mark_actioned(WORK_QUEUE, last_message)

            if last_message.mtype == START_HPO:
                info('HPO')
                msg = last_message.message
                hpo = WorkScheduler(**msg.get('kwargs'))
                hpo.run()

        if self.broker:
            self.broker.stop()

    def execute(self, message: Message):
        msg = message.message

        script = msg.get('script')
        args = msg.get('args', list())
        env = msg.get('env', dict())
        env['MQ_CLIENT'] = self.uri

        if script is None:
            error(f'work item without script!')
        else:
            info(f'starting work process (cmd: {script} {" ".join(args)})')

            command = [script] + args
            process = subprocess.Popen(command, env=env)
            return_code = process.wait()
            info(f'finished work item (rc: {return_code})')

        self.client.mark_actioned(WORK_QUEUE, uid=message.uid)
