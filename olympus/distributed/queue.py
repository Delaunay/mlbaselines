from dataclasses import dataclass
from datetime import datetime


@dataclass
class Agent:
    uid: int
    time: datetime
    agent: str
    heartbeat: datetime


@dataclass
class Message:
    uid: int
    time: datetime
    mtype: int
    read: bool
    read_time: datetime
    actioned: bool
    actioned_time: datetime
    message: str

    def __repr__(self):
        return f"""Message({self.uid}, {self.time}, {self.mtype}, {self.read}, """ +\
            f"""{self.read_time}, {self.actioned}, {self.actioned_time}, {self.message})"""


class MessageQueue:
    def add_handler(self, mtype, handler):
        """Add a parser for a message type

        Parameters
        ----------
        mtype: int
            message type

        handler: Callable[[Json], ...]
            message parser

        """
        raise NotImplementedError()

    def enqueue(self, name, message, mtype=0):
        """Insert a new message inside the queue

        Parameters
        ----------
        name: str
            Message queue namespace

        message: str
            message to insert

        mtype: int
            message type

        """
        raise NotImplementedError()

    def get_unactioned(self, name):
        """Return unactioned messages

        Parameters
        ----------
        name: str
            message queue namespace used to get unactioned messages
        """
        raise NotImplementedError()

    def dequeue(self, name):
        """Remove oldest message from the queue i.e mark is as read

        Parameters
        ----------
        name: str
            Queue namespace to pop message from

        """
        raise NotImplementedError()

    def mark_actioned(self, name, message: Message = None, uid: int = None):
        """Mark a message as actioned

        Parameters
        ----------
        name: str
            Message queue namespace

        message: Optional[str]
            message object to update

        uid: Optional[int]
            uid of the message to update

        """
        raise NotImplementedError()

    def push(self, name, message, mtype=0):
        return self.enqueue(name, message, mtype)

    def pop(self, name):
        return self.dequeue(name)

    def unread_count(self, name):
        raise NotImplementedError()

    def unactioned_count(self, name):
        raise NotImplementedError()

    def actioned_count(self, name):
        raise NotImplementedError()

    def read_count(self, name):
        raise NotImplementedError()

    def reset_queue(self, name):
        raise NotImplementedError()
