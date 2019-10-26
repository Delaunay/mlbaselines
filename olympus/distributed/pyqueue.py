import json
import datetime
import multiprocessing

from multiprocessing.sharedctypes import Value
from ctypes import Structure, c_int, c_bool, c_char, c_double

from olympus.utils import warning
from olympus.distributed.queue import Message, MessageQueue

MAX_SIZE = 1024

_manager = None
_queues = None
_count = None
_table = None


def manager():
    global _manager, _count, _table, _queues

    if _manager is None:
        _manager = multiprocessing.Manager()
        _manager.__enter__()
        _count = Value(c_int, 0)
        _table = _manager.dict()
        _queues = _manager.dict()

    return _manager


def make_queue(name):
    global _queues
    queue = _queues.get(name)

    if queue is None:
        queue = manager().list()
        _queues[name] = queue

    return queue


def _get_uid():
    global _count
    v = _count.value
    with _count.get_lock():
        _count.value += 1
    return v


class MessageStruct(Structure):
    _fields_ = [
        ('uid', c_int),
        ('time', c_double),
        ('mtype', c_int),
        ('read', c_bool),
        ('read_time', c_double),
        ('actioned', c_bool),
        ('actioned_time', c_double),
        ('message', c_char * MAX_SIZE),
    ]

    def __repr__(self):
        return f"""CMessage({self.uid}, {self.time}, {self.mtype}, {self.read}, """ +\
            f"""{self.read_time}, {self.actioned}, {self.actioned_time}, {self.message})"""


def to_timestamp(v):
    if v is None:
        return 0
    return datetime.datetime.timestamp(v)


def make_struct(uid, time, mtype, read, read_time, actioned, actioned_time, message):
    m = MessageStruct(
        uid, to_timestamp(time), mtype,
        read, to_timestamp(read_time),
        actioned, to_timestamp(actioned_time),
        json.dumps(message).encode('utf8')
    )
    print(m)
    return m


def struct_to_message(struct: MessageStruct):
    m = Message(
        struct.uid, struct.time, struct.mtype,
        struct.read, struct.read_time,
        struct.actioned, struct.actioned_time,
        json.loads(struct.message)
    )
    print(m)
    return m


class PythonBroker:
    def __init__(self):
        self.manager = manager()

    def start(self):
        pass

    def new_queue(self, name):
        make_queue(name)

    def stop(self):
        global  _manager

        manager().__exit__(None, None, None)
        _manager = None


class PythonQueue(MessageQueue):
    def __init__(self):
        self.manager = manager()
        self.queue = _queues
        self.message_handlers = {
            0: self.no_handler
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @staticmethod
    def no_handler(msg):
        return msg

    def add_handler(self, mtype, handler):
        """See `~mlbaselines.distributed.queue.MessageQueue`"""
        self.message_handlers[mtype] = handler

    def dequeue(self, name):
        result = self.queue.get(name).pop(1)
        if result is None:
            return None

        m = struct_to_message(result)
        m.read = True
        m.read_time = datetime.datetime.now()
        return m

    def enqueue(self, name, message, mtype=0):
        s = make_struct(
            uid=_get_uid(), time=datetime.datetime.now(), mtype=mtype,
            read=False, read_time=None,
            actioned=False, actioned_time=None,
            message=message
        )
        return self.queue.get(name).append(s)

    def get_unactioned(self, name):
        return None

    def mark_actioned(self, name, message: Message = None, uid: int = None):
        return True

    def unread_count(self, name):
        return self.queue.get(name).qsize()

    def unactioned_count(self, name):
        return None

    def actioned_count(self, name):
        return None

    def read_count(self, name):
        return None

    def reset_queue(self, name):
        warning('Python Queues do not support reset')
        return None

    def agent_count(self):
        agents = self.queue.get('system')
        if agents:
            return len(agents)
        return 0
