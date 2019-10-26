import os
import json
import psycopg2
import shutil
import signal
import subprocess
import time
import traceback
import threading

from multiprocessing import Process, Manager
from olympus.utils import parse_uri, debug, info, error
from olympus.distributed.queue import Message, MessageQueue


VERSION = '19.1.1'

COCKROACH_HASH = {
    'posix': '051b9f3afd3478b62e3fce0d140df6f091b4a1e4ef84f05c3f1c3588db2495fa',
    'macos': 'ec1fe3dfb55c67b74c3f04c15d495a55966b930bb378b416a5af5f877fb093de'
}

_base = os.path.dirname(os.path.realpath(__file__))

COCKROACH_BIN = {
    'posix': f'{_base}/cockroach/cockroach_linux',
    'macos': f'{_base}/cockroach/cockroach_macos'
}


def track_schema(clients):
    permissions = []
    for client in clients:
        permissions.append(f'GRANT ALL ON DATABASE track TO {client};')
    permissions = '\n'.join(permissions)

    return f"""
    CREATE DATABASE IF NOT EXISTS track;
    SET DATABASE = track;
    {permissions}
    CREATE TABLE IF NOT EXISTS track.projects (
        uid             BYTES PRIMARY KEY,
        name            STRING,
        description     STRING,
        metadata        JSONB,
        trial_groups    BYTES[],
        trials          BYTES[]
    );
    CREATE TABLE IF NOT EXISTS track.trial_groups (
        uid         BYTES PRIMARY KEY,
        name        STRING,
        description STRING,
        metadata    JSONB,
        trials      BYTES[],
        project_id  BYTES
    );
    CREATE TABLE IF NOT EXISTS track.trials (
        uid         BYTES,
        hash        BYTES,
        revision    SMALLINT,
        name        STRING,
        description STRING,
        tags        JSONB,
        version     BYTES,
        group_id    BYTES,
        project_id  BYTES,
        parameters  JSONB,
        metadata    JSONB,
        metrics     JSONB,
        chronos     JSONB,
        status      JSONB,
        errors      JSONB,

        PRIMARY KEY (hash, revision)
    );""".encode('utf8')


def message_queue_schema(clients, name):
    """Create a message queue table

    uid          : message uid to update messages
    time         : timestamp when the message was created
    read         : was the message read
    read_time    : when was the message read
    actioned     : was the message actioned
    actioned_time: when was the message actioned
    message      : the message
    """
    permissions = []
    for client in clients:
        permissions.append(f'GRANT ALL ON DATABASE queue_{name} TO {client};')
    permissions = '\n'.join(permissions)

    return f"""
    CREATE DATABASE IF NOT EXISTS queue_{name};
    SET DATABASE = queue_{name};
    {permissions}
    CREATE TABLE IF NOT EXISTS queue_{name}.messages (
        uid             SERIAL      PRIMARY KEY,
        time            TIMESTAMP   DEFAULT current_timestamp(),
        mtype           INT,
        read            BOOLEAN,
        read_time       TIMESTAMP,
        actioned        BOOLEAN,
        actioned_time   TIMESTAMP,
        message         JSONB
    );
    CREATE DATABASE IF NOT EXISTS queue;
    GRANT ALL ON DATABASE queue TO {client};
    CREATE TABLE IF NOT EXISTS queue.system (
        uid             SERIAL      PRIMARY KEY,
        time            TIMESTAMP   DEFAULT current_timestamp(),
        agent           JSONB,
        heartbeat       TIMESTAMP   DEFAULT current_timestamp()
    );
    CREATE INDEX IF NOT EXISTS messages_index
    ON queue_{name}.messages  (
        read        DESC, 
        time        DESC,
        actioned    DESC,
        mtype       ASC
    );
    """


class CockRoachDB:
    """ cockroach db is a highly resilient database that allow us to remove
    the Master in a traditional distributed setup.

    This spawn a cockroach node that will store its data in `location`
    """

    def __init__(self, location, addrs, join=None, clean_on_exit=True, schema=None):
        self.location = location

        logs = f'{location}/logs'
        temp = f'{location}/tmp'
        external = f'{location}/extern'
        store = location

        os.makedirs(logs, exist_ok=True)
        os.makedirs(temp, exist_ok=True)
        os.makedirs(external, exist_ok=True)

        self.location = location
        self.addrs = addrs
        self.bin = COCKROACH_BIN.get(os.name)

        if self.bin is None:
            raise RuntimeError('Your OS is not supported')

        if not os.path.exists(self.bin):
            info('Using system binary')
            self.bin = 'cockroach'
        else:
            hash = COCKROACH_HASH.get(os.name)

        self.arguments = [
            'start', '--insecure',
            f'--listen-addr={addrs}',
            f'--external-io-dir={external}',
            f'--store={store}',
            f'--temp-dir={temp}',
            f'--log-dir={logs}',
            f'--pid-file={location}/cockroach_pid'
        ]

        if join is not None:
            self.arguments.append(f'--join={join}')

        self.manager: Manager = Manager()
        self.properties = self.manager.dict()
        self.properties['running'] = False
        self.clean_on_exit = clean_on_exit
        self._process: Process = None
        self.cmd = None
        self.schema = schema

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
                while self.properties.get('nodeID') is None and self._process.is_alive():
                    time.sleep(0.01)

            self.properties['db_pid'] = int(open(f'{self.location}/cockroach_pid', 'r').read())
            self._setup()
        except Exception as e:
            error(traceback.format_exc(e))

    def _setup(self, client='track_client'):
        out = subprocess.check_output(
            f'{self.bin} user set {client} --insecure --host={self.addrs}', shell=True)
        debug(out.decode('utf8').strip())

        if self.schema is not None:
            if callable(self.schema):
                self.schema = self.schema(client)

            out = subprocess.check_output(
                f'{self.bin} sql --insecure --host={self.addrs}', input=self.schema, shell=True)
            debug(out.decode('utf8').strip())

    def new_queue(self, name, client='default_user', clients=None):
        """Create a new queue

        Parameters
        ----------
        name: str
            create a new queue named `name`

        client: str
            client name to use

        clients: str
            create permission for all the clients
        """
        # set client
        out = subprocess.check_output(
            f'{self.bin} user set {client} --insecure --host={self.addrs}', shell=True)
        debug(out.decode('utf8').strip())

        if clients is None:
            clients = []

        clients.append(client)

        statement = message_queue_schema(clients, name)
        if isinstance(statement, str):
            statement = statement.encode('utf8')

        out = subprocess.check_output(
            f'{self.bin} sql --insecure --host={self.addrs}', input=statement, shell=True)
        debug(out.decode('utf8').strip())

    def stop(self):
        self.properties['running'] = False
        self._process.terminate()

        os.kill(self.properties['db_pid'], signal.SIGTERM)

        if self.clean_on_exit:
            shutil.rmtree(self.location)

    def wait(self):
        while True:
            time.sleep(0.01)

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if exc_type is not None:
            raise exc_type

    def parse(self, properties, line):
        if line[0] == '*':
            return
        try:
            a, b = line.split(':', maxsplit=1)
            properties[a.strip()] = b.strip()

        except Exception as e:
            print(e, line, end='\n')
            print(traceback.format_exc())
            raise RuntimeError(f'{line} (cmd: {self.cmd})')

    # properties that are populated once the server has started
    @property
    def node_id(self):
        return self.properties.get('nodeID')

    @property
    def status(self):
        return self.properties.get('status')

    @property
    def sql(self):
        return self.properties.get('sql')

    @property
    def client_flags(self):
        return self.properties.get('client flags')

    @property
    def webui(self):
        return self.properties.get('webui')

    @property
    def build(self):
        return self.properties.get('build')


def start_message_queue(name, location, addrs, join=None, clean_on_exit=True):
    cockroach = CockRoachDB(location, addrs, join, clean_on_exit, schema=None)
    return cockroach


class AgentMonitor(threading.Thread):
    def __init__(self, agent, wait_time=60):
        threading.Thread.__init__(self)
        self.stopped = threading.Event()
        self.wait_time = wait_time
        self.agent = agent
        self.cursor = agent.cursor

    def stop(self):
        """Stop monitoring."""
        self.stopped.set()
        self.join()

    def run(self):
        """Run the trial monitoring every given interval."""
        while not self.stopped.wait(self.wait_time):
            self.update_heartbeat()

    def update_heartbeat(self):
        self.cursor.execute(f"""
        UPDATE 
            queue.system
        SET 
            heartbeat = current_timestamp()
        WHERE
            uid = %s
        """, (self.agent.agent_id,))


class CKMQClient(MessageQueue):
    """Simple cockroach db queue client

    Parameters
    ----------
    uri: str
        cockroach://192.168.0.10:8123
    """

    def __init__(self, uri, name=None):
        uri = parse_uri(uri)

        self.con = psycopg2.connect(
            user=uri.get('username', 'default_user'),
            password=uri.get('password', 'mq_password'),
            # sslmode='require',
            # sslrootcert='certs/ca.crt',
            # sslkey='certs/client.maxroach.key',
            # sslcert='certs/client.maxroach.crt',
            port=uri['port'],
            host=uri['address']
        )
        self.con.set_session(autocommit=True)
        self.cursor = self.con.cursor()
        self.name = name
        self.agent_id = None
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
        self.cursor.execute(f"""
        INSERT INTO
            queue.system (agent)
        VALUES
            (%s)
        RETURNING uid
        """, (json.dumps(agent_name),))
        return self.cursor.fetchone()[0]

    def _update_heartbeat(self):
        self.cursor.execute(f"""
        UPDATE queue.system SET
            heartbeat = current_timestamp()
        WHERE
            uid = %s
        """, (self.agent_id,))

    def _remove(self):
        self.cursor.execute(f'DELETE FROM queue.system WHERE uid = %s', (self.agent_id,))

    @staticmethod
    def no_handler(msg):
        return msg

    def add_handler(self, mtype, handler):
        """See `~mlbaselines.distributed.queue.MessageQueue`"""
        self.message_handlers[mtype] = handler

    def enqueue(self, name, message, mtype=0):
        """See `~mlbaselines.distributed.queue.MessageQueue`"""
        self.cursor.execute(f"""
        INSERT INTO  
            queue_{name}.messages (mtype, read, actioned, message)
        VALUES
            (%s, %s, %s, %s)
        """, (mtype, False, False, json.dumps(message)))

    def _parse(self, result):
        if result is None:
            return None

        mtype = result[2]
        parser = self.message_handlers.get(mtype, self.no_handler)
        return Message(
            result[0],
            result[1],
            mtype,
            result[3],
            result[4],
            result[5],
            result[6],
            parser(result[7]),
        )

    def get_unactioned(self, name):
        """See `~mlbaselines.distributed.queue.MessageQueue`"""
        self.cursor.execute(f"""
        SELECT 
            * 
        FROM 
            queue_{name}.messages 
        WHERE
            actioned = false
        """)
        rows = self.cursor.fetchall()
        return [self._parse(r) for r in rows]

    def dump(self, name):
        self.cursor.execute(f'SELECT *  FROM queue_{name}.messages')

        rows = self.cursor.fetchall()
        for row in rows:
            print(self._parse(row))

    def dequeue(self, name):
        """See `~mlbaselines.distributed.queue.MessageQueue`"""

        self.cursor.execute(f"""
        UPDATE queue_{name}.messages SET 
            (read, read_time) = (true, current_timestamp())
        WHERE 
            read = false
        ORDER BY
            time
        LIMIT 1
        RETURNING *
        """)

        return self._parse(self.cursor.fetchone())

    def mark_actioned(self, name, message: Message = None, uid: int = None):
        """See `~mlbaselines.distributed.queue.MessageQueue`"""

        if isinstance(message, Message):
            uid = message.uid

        self.cursor.execute(f"""
        UPDATE queue_{name}.messages SET 
            (actioned, actioned_time) = (true, current_timestamp())
        WHERE 
            uid = %s
        RETURNING *
        """, (uid, ))

        return self._parse(self.cursor.fetchone())

    def unread_count(self, name):
        self.cursor.execute(f"""
        SELECT 
            COUNT(*)
        FROM 
            queue_{name}.messages
        WHERE 
            read = false
        """)
        return self.cursor.fetchone()[0]

    def unactioned_count(self, name):
        self.cursor.execute(f"""
        SELECT 
            COUNT(*)
        FROM 
            queue_{name}.messages
        WHERE 
            actioned = false
        """)
        return self.cursor.fetchone()[0]

    def read_count(self, name):
        self.cursor.execute(f"""
        SELECT 
            COUNT(*)
        FROM 
            queue_{name}.messages
        WHERE 
            read = true
        """)
        return self.cursor.fetchone()[0]

    def actioned_count(self, name):
        self.cursor.execute(f"""
        SELECT 
            COUNT(*)
        FROM 
            queue_{name}.messages
        WHERE 
            actioned = true
        """)
        return self.cursor.fetchone()[0]

    def reset_queue(self, name):
        """Resume queue from previous statement

        Returns
        --------
        returns all the restored messages

        """
        self.cursor.execute(f"""
        UPDATE queue_{name}.messages 
            SET 
                (read, read_time) = (false, null)
            WHERE 
                actioned = false
        RETURNING *
        """)

        rows = self.cursor.fetchall()
        records = []
        for row in rows:
            records.append(self._parse(row))

        return records
