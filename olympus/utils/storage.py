import datetime
import os
from typing import Callable

import torch


class StateStorage:
    def __init__(self, folder, time_buffer=5 * 60):
        # typically root/task_name/experiment_name/trial_id
        self.folder = folder
        self.time_buffer = time_buffer
        self.last_save = datetime.datetime.utcnow()
        os.makedirs(self.folder, exist_ok=True)
        self.cache = set()

    def _file(self, filename):
        return f'{self.folder}/{filename}.state'

    def open(self, filename, mode):
        return open(self._file(filename), mode)

    def write(self, filename, data):
        return self.open(filename, 'w').write(data)

    def read(self, filename):
        return self.open(filename, 'r').read()

    def exits(self, filename):
        return os.path.exists(self._file(filename))

    def save(self, filename, state):
        if (datetime.datetime.utcnow() - self.last_save).total_seconds() > self.time_buffer:

            path = self._file(filename)

            dirname = os.path.dirname(path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

            torch.save(state, path)
            self.last_save = datetime.datetime.utcnow()

    def load(self, filename, device=None):
        """

        Parameters
        ----------
        filename: str
            file to load the state from

        device: torch.device
            it indicates the location where all tensors should be loaded.
        """
        return torch.load(self._file(filename), map_location=device)


if __name__ == '__main__':
    storage = StateStorage('/fast/states')

    with storage.open('my_model_state1', 'w') as file:
        file.write('1234')

    storage.write('my_model_state2', '11234')
    print('done')

    print(storage.read('my_model_state2'))

    storage.save('my_state1', {
        'abc', 10
    })

    print(storage.load('my_state1'))

