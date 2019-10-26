import os
import torch


class StateStorage:
    def __init__(self, folder):
        self.folder = folder
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
        torch.save(state, self._file(filename))

    def load(self, filename):
        return torch.load(self._file(filename))


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

