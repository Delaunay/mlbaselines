from datetime import datetime
import json

from olympus.utils.stat import StatStream


class ChronoContext:
    def __init__(self, stream):
        self.start = None
        self.end = None
        self.stream = stream

    def __enter__(self):
        self.start = datetime.utcnow()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = datetime.utcnow()
        self.stream += (self.end - self.start).total_seconds()


class Chrono:
    def __init__(self):
        self.chronos = dict()
        self.start = datetime.utcnow()

    def time(self, name):
        if name not in self.chronos:
            self.chronos[name] = StatStream(drop_first_obs=0)

        return ChronoContext(self.chronos[name])

    def show(self):
        data = {}

        for k, v in self.chronos.items():
            if v.count > 0:
                data[k] = v.to_json()

        print(json.dumps(data, indent=2))
