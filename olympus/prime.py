import time

from olympus.distributed import make_message_broker, make_message_client
from olympus.hpo import WORK_QUEUE, RESULT_QUEUE
from olympus.hpo import START_BROKER, START_HPO


class PrimeMonitor:
    def __init__(self, message_queue, world_size=None):
        self.message_queue = message_queue
        self.broker = make_message_broker(self.message_queue)
        self.broker.start()
        self.broker.new_queue(WORK_QUEUE)
        self.broker.new_queue(RESULT_QUEUE)
        self.client = make_message_client(self.message_queue)
        self.world_size = world_size

    def _show(self, queue, messages):
        if messages:
            header = f'{queue} (len: {len(messages)})'
            print(header)
            print('-' * len(header))
            for m in messages:
                print(f'    - {m}')
            print('-' * 80)

    def restore_session(self):
        restored_messages = self.client.reset_queue(WORK_QUEUE)
        self._show(WORK_QUEUE, restored_messages)

        restored_messages = self.client.reset_queue(RESULT_QUEUE)
        self._show(RESULT_QUEUE, restored_messages)

    def queue_hpo(self):
        self.client.push(
            WORK_QUEUE,
            mtype=START_HPO,
            message={
                'kwargs': {
                    'message_queue': self.message_queue
                }
            })

    def queue_broker(self):
        self.client.push(
            WORK_QUEUE,
            mtype=START_BROKER,
            message={
                'kwargs': {
                    'uri': self.message_queue
                }
            })

    def agent_count(self):
        # self.client.cursor.execute('SELECT COUNT(*) FROM queue.system')
        return self.client.agent_count()

    def wait(self, poll_time=0.01):
        sleep = 0
        while self.agent_count() > 0:
            time.sleep(poll_time)
            sleep += poll_time

            if sleep > 10:
                self.show_system()
                sleep = 0

    def shutdown(self):
        self.broker.stop()

    def show_system(self):
        # self.client.cursor.execute('SELECT * FROM queue.system')
        #
        # def parse(r):
        #     return Agent(r[0], r[1], r[2], r[3])
        #
        # agents = [parse(r) for r in self.client.cursor.fetchall()]
        agents = self.client.agents()

        print(f'.{"-" * 12}-{"-" * 12}-{"-" * 28}.')
        print(f'|{" " * 12} {"System":12} {" " * 28}|')
        print(f'|{"-" * 12}-{"-" * 12}-{"-" * 28}|')
        print(f'| {"uid":>10} | {"name":>10} | {"heartbeat":>26} |')
        print(f'|{"-" * 12}+{"-" * 12}+{"-" * 28}|')
        for agent in agents:
            print(f'| {str(agent.uid)[-10:]} | {agent.agent:>10} | {str(agent.heartbeat - agent.time):>26} |')
        print(f'\'{"-" * 12}-{"-" * 12}-{"-" * 28}\'')
        print()
