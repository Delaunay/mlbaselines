import datetime
import os
import pymongo
import shutil
import signal
import subprocess
import time
import traceback
import threading

from multiprocessing import Process, Manager
from olympus.utils import parse_uri, info, error
from olympus.distributed.queue import Message, MessageQueue, Agent

_base = os.path.dirname(os.path.realpath(__file__))

MONGODB_BIN = {
    'posix': f'{_base}/mongo/mongo_linux',
    'macos': f'{_base}/mongo/mongo_macos'
}


class MongoDB:
    def __init__(self, address, port, location, clean_on_exit=True):
        self.location = location
        self.data_path = f'{self.location}/db'
        self.pid_file = f'{self.location}/pid'

        os.makedirs(self.data_path, exist_ok=True)

        self.address = address
        self.port = port
        self.location = location
        self.bin = 'mongod'

        if self.bin is None:
            raise RuntimeError('Your OS is not supported')

        if not os.path.exists(self.bin):
            info('Using system binary')
            self.bin = 'mongod'

        self.arguments = [
            '--dbpath', self.data_path,
            '--wiredTigerCacheSizeGB', '1',
            '--port', str(port),
            '--bind_ip', address,
            '--pidfilepath', self.pid_file
        ]

        self.manager: Manager = Manager()
        self.properties = self.manager.dict()
        self.properties['running'] = False
        self.clean_on_exit = clean_on_exit
        self._process: Process = None
        self.cmd = None

    def _start(self, properties):
        kwargs = dict(
            args=' '.join([self.bin] + self.arguments),
            stdout=subprocess.PIPE,
            bufsize=1,
            stderr=subprocess.STDOUT
        )
        self.cmd = kwargs['args']

        with subprocess.Popen(**kwargs, shell=True) as proc:
            try:
                properties['running'] = True
                properties['pid'] = proc.pid

                while properties['running']:
                    if proc.poll() is None:
                        line = proc.stdout.readline().decode('utf-8')
                        if line:
                            self.parse(properties, line)
                    else:
                        properties['running'] = False
                        properties['exit'] = proc.returncode

            except Exception:
                error(traceback.format_exc())

    def start(self, wait=True):
        try:
            self._process = Process(target=self._start, args=(self.properties,))
            self._process.start()

            # wait for all the properties to be populated
            if wait:
                while self.properties.get('ready') is None:
                    time.sleep(0.01)

            self.properties['db_pid'] = int(open(self.pid_file, 'r').read())
            self._setup()

        except Exception as e:
            error(traceback.format_exc(e))

    def _setup(self, client='track_client'):
        pass

    def new_queue(self, name, client='default_user', clients=None):
        client = pymongo.MongoClient(
            host=self.address,
            port=self.port)

        queues = client.queues
        queue = queues[name]
        queue.create_index([
            ('time', pymongo.DESCENDING),
            ('mtype', pymongo.DESCENDING),
            ('read', pymongo.DESCENDING),
            ('actioned', pymongo.DESCENDING)
        ])

    def stop(self):
        self.properties['running'] = False
        self._process.terminate()

        try:
            os.kill(self.properties['db_pid'], signal.SIGTERM)
        except ProcessLookupError:
            pass

        if self.clean_on_exit:
            shutil.rmtree(self.location)

    def wait(self):
        while self._process.is_alive():
            time.sleep(0.01)

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if exc_type is not None:
            raise exc_type

    def parse(self, properties, line):
        line = line.strip()

        if line.endswith(f'waiting for connections on port {self.port}'):
            properties['ready'] = True


def start_message_queue(location, addrs, join=None, clean_on_exit=True):
    cockroach = MongoDB(location, addrs, join, clean_on_exit, schema=None)
    return cockroach


class AgentMonitor(threading.Thread):
    def __init__(self, agent, wait_time=60):
        threading.Thread.__init__(self)
        self.stopped = threading.Event()
        self.wait_time = wait_time
        self.agent = agent
        self.client = agent.client

    def stop(self):
        """Stop monitoring."""
        self.stopped.set()
        self.join()

    def run(self):
        """Run the trial monitoring every given interval."""
        while not self.stopped.wait(self.wait_time):
            self.update_heartbeat()

    def update_heartbeat(self):
        self.client.queues.system.update_one(
            {'_id': self.agent.agent_id},
            {'$set': {
                'heartbeat': datetime.datetime.utcnow()
                }
            })


class MongoClient(MessageQueue):
    """Simple cockroach db queue client

    Parameters
    ----------
    uri: str
        mongodb://192.168.0.10:8123
    """

    def __init__(self, uri, name='worker'):
        uri = parse_uri(uri)
        self.name = name
        self.client = pymongo.MongoClient(host=uri['address'], port=int(uri['port']))
        self.heartbeat_monitor = AgentMonitor(self, wait_time=60)
        self.message_handlers = {
            0: self.no_handler
        }

    def __enter__(self):
        self.agent_id = self._register_agent(self.name)
        self.heartbeat_monitor.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.heartbeat_monitor.stop()
        self._remove()

    def _register_agent(self, agent_name):
        rc = self.client.queues.system.insert_one(
            {
                'time': datetime.datetime.utcnow(),
                'agent': agent_name,
                'heartbeat': datetime.datetime.utcnow()
            }
        ).inserted_id
        return rc

    def _update_heartbeat(self):
        return self.client.update_one({'_id': self.agent_id}, {
            'heartbeat': datetime.datetime.utcnow()
        })

    def _remove(self):
        self.client.queues.system.remove({'_id': self.agent_id})

    @staticmethod
    def no_handler(msg):
        return msg

    def add_handler(self, mtype, handler):
        """See `~mlbaselines.distributed.queue.MessageQueue`"""
        self.message_handlers[mtype] = handler

    def enqueue(self, name, message, mtype=0):
        """See `~mlbaselines.distributed.queue.MessageQueue`"""
        return self.client.queues[name].insert_one({
            'time': datetime.datetime.utcnow(),
            'mtype': mtype,
            'read': False,
            'read_time': None,
            'actioned': False,
            'actioned_time': None,
            'message': message
        })

    def _parse(self, result):
        if result is None:
            return None

        mtype = result['mtype']
        parser = self.message_handlers.get(mtype, self.no_handler)
        return Message(
            result['_id'],
            result['time'],
            mtype,
            result['read'],
            result['read_time'],
            result['actioned'],
            result['actioned_time'],
            parser(result['message']),
        )

    def get_unactioned(self, name):
        """See `~mlbaselines.distributed.queue.MessageQueue`"""
        return [self._parse(msg) for msg in self.client.queues[name].find({'actioned': False})]

    def dump(self, name):
        rows = self.client.queues[name].find()
        for row in rows:
            print(self._parse(row))

    def dequeue(self, name):
        """See `~mlbaselines.distributed.queue.MessageQueue`"""
        msg = self.client.queues[name].find_one_and_update(
            {'read': False},
            {'$set': {
                'read': True, 'read_time': datetime.datetime.utcnow()}
            },
            return_document=pymongo.ReturnDocument.AFTER
        )
        return self._parse(msg)

    def mark_actioned(self, name, message: Message = None, uid: int = None):
        """See `~mlbaselines.distributed.queue.MessageQueue`"""
        if isinstance(message, Message):
            uid = message.uid

        msg = self.client.queues[name].find_one_and_update(
            {'_id': uid},
            {'$set': {
                'actioned': True,
                'actioned_time': datetime.datetime.utcnow()}
            },
            return_document=pymongo.ReturnDocument.AFTER
        )

        return self._parse(msg)

    def unread_count(self, name):
        return self.client.queues[name].count({'read': False})

    def unactioned_count(self, name):
        return self.client.queues[name].count({'actioned': False})

    def read_count(self, name):
        return self.client.queues[name].count({'read': True})

    def actioned_count(self, name):
        return self.client.queues[name].count({'actioned': True})

    def agent_count(self):
        return self.client.queues.system.count()

    def agents(self):
        agents = self.client.queues.system.find()
        results = []

        for agent in agents:
            agent['uid'] = agent['_id']
            agent.pop('_id')

            results.append(Agent(**agent))

        return results

    def reset_queue(self, name):
        """Resume queue from previous statement

        Returns
        --------
        returns all the restored messages

        """

        msgs = self.client.queues[name].find({'actioned': False, 'read':  True})
        rc = self.client.queues[name].update(
            {'actioned': False},
            {'$set': {
                'read': False, 'read_time': None}
            }
        )

        items = []
        for msg in msgs:
            items.append(self._parse(msg))
        return items


def start_mongod():
    from argparse import ArgumentParser
    import os

    parser = ArgumentParser()
    parser.add_argument('--address', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=8123)
    parser.add_argument('--loc', type=str, default=os.getcwd())
    args = parser.parse_args()

    print(args.port)
    server = MongoDB(args.address, args.port, args.loc, False)

    server.start()
