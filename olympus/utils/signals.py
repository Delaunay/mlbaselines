import signal
import sys
import time
import atexit

from .log import warning


class SignalHandler:
    def __init__(self):
        signal.signal(signal.SIGINT, self._sigint)
        signal.signal(signal.SIGTERM, self._sigterm)
        atexit.register(self.atexit)

    def _sigterm(self, signum, frame):
        self.sigterm(signum, frame)
        sys.exit(1)

    def _sigint(self, signum, frame):
        self.sigint(signum, frame)
        sys.exit(1)

    def sigterm(self, signum, frame):
        raise NotImplementedError()

    def sigint(self, signum, frame):
        raise NotImplementedError()

    def atexit(self):
        pass


class Protected(object):
    def __init__(self):
        self.signal_received = None
        self.handlers = dict()
        self.start = 0

    def __enter__(self):
        self.signal_received = False
        self.start = time.time()
        self.handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self.handler)
        self.handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self.handler)

    def handler(self, sig, frame):
        warning(f'Delaying signal {sig} to finish operations')
        self.signal_received = (sig, frame)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.handlers[signal.SIGINT])
        signal.signal(signal.SIGTERM, self.handlers[signal.SIGTERM])

        if self.signal_received:
            warning(f'Termination was delayed by {time.time() - self.start:.4f} s')
            handler = self.handlers[self.signal_received[0]]

            if callable(handler):
                handler(*self.signal_received)
