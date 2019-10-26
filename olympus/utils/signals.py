import signal
import sys
import atexit


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

