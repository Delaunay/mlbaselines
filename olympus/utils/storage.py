from datetime import datetime
import os
import io
import tempfile
import torch

from olympus.utils import info
from olympus.utils.options import options


class BaseStorage:
    def load(self, *args, **kwargs):
        pass

    def safe_load(self, *args, **kwargs):
        pass

    def __init__(self, *args, **kwargs):
        pass

    def set_base(self, *args, **kwargs):
        pass

    def show_memory_usage(self):
        return {}

    def garbage_collect_in_memory(self, *args, **kwargs):
        pass

    def garbage_collect_on_disk(self, *args, **kwargs):
        pass

    def garbage_collect(self, *args, **kwargs):
        pass

    def open(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass

    def read(self, *args, **kwargs):
        pass

    def exits(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass


def NoStorage(*args, **kwargs):
    return BaseStorage(*args, **kwargs)


class StateStorage(BaseStorage):
    Kio = 1024
    Mio = 1024 * 1024
    USE_IN_MEMORY_CACHE = False

    def __init__(self, folder=options('state.storage', '/tmp'),
                 time_buffer=options('state.storage.time', 5 * 60, type=int)):
        # typically root/task_name/experiment_name/trial_id
        self.folder = None
        self.set_base(folder)

        self.time_buffer = time_buffer
        self.last_save = datetime.utcnow()

        self.cache = dict()
        self.in_memory = 0
        self.on_disk = 0
        self.on_disk_files = dict()

    def set_base(self, folder):
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)

    def show_memory_usage(self):
        return {
            'on_disk': self.on_disk / StateStorage.Mio,
            'on_disk_file_count': len(self.on_disk_files),
            'in_memory': self.in_memory / StateStorage.Mio
        }

    def garbage_collect_in_memory(self, gc_time):
        now = datetime.utcnow()
        old = self.in_memory
        to_be_deleted = []

        for name, (buffer, save_time) in self.cache.items():
            if (now - save_time).total_seconds > gc_time:
                to_be_deleted.append(name)

        for name in to_be_deleted:
            self._pop_from_cache(name)

        new = self.in_memory
        freed = old - new
        return freed

    def garbage_collect_on_disk(self, gc_time):
        now = datetime.utcnow()
        old = self.on_disk
        to_be_deleted = []

        for path, (size, save_time) in self.on_disk_files.items():
            if (now - save_time).total_seconds > gc_time:
                to_be_deleted.append(path)

        for path in to_be_deleted:
            self._pop_from_disk(path)

        new = self.on_disk
        freed = old - new
        return freed

    def garbage_collect(self, gc_time):
        freed = 0
        freed += self.garbage_collect_in_memory(gc_time)
        freed += self.garbage_collect_on_disk(gc_time)

        return freed

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
        from olympus.utils import info

        path = self._file(filename)
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # Writes the state inside a buffer
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer = buffer.getbuffer()

        # if it has been a while write it to disk
        elapsed_time = (datetime.utcnow() - self.last_save).total_seconds()
        if elapsed_time > self.time_buffer:
            # write it inside a temporary file
            fd, name = tempfile.mkstemp('state', dir=self.folder)

            file = os.fdopen(fd, 'wb')
            file.write(buffer)
            file.close()

            # move temporary file to right spot
            os.rename(name, path)
            info(f'State was written to {path}')

            # Remove from cache it is in
            self._pop_from_cache(filename)
            self._insert_disk(filename, buffer.nbytes)
            self.last_save = datetime.utcnow()
            return True
        else:
            info(f'({elapsed_time:.2f} > {self.time_buffer}) skipping checkpoint')

        self._insert_cache(filename, buffer)
        return False

    def _insert_disk(self, key, size):
        if key in self.on_disk_files:
            self._pop_from_disk(key)

        self.on_disk_files[key] = (size, datetime.utcnow())
        self.on_disk += size

    def _pop_from_disk(self, key):
        size, _ = self.on_disk_files.pop(key, (None, None))
        if size:
            self.on_disk -= size

    def _insert_cache(self, key, buffer):
        if StateStorage.USE_IN_MEMORY_CACHE:
            if key in self.cache:
                self._pop_from_cache(key)

            self.cache[key] = (buffer, datetime.utcnow())
            self.in_memory += buffer.getbuffer().nbytes

    def _pop_from_cache(self, key):
        buffer, _ = self.cache.pop(key, (None, None))
        if buffer:
            self.in_memory -= buffer.getbuffer().nbytes

        return buffer

    def load(self, filename, device=None):
        """

        Parameters
        ----------
        filename: str
            file to load the state from

        device: torch.device
            it indicates the location where all tensors should be loaded.
        """
        buffer = self._pop_from_cache(filename)

        if buffer is None:
            buffer = self._file(filename)

        return torch.load(buffer, map_location=device)

    def safe_load(self, name, device):
        """Handles a few common exceptions for you and returns None if a file is not found"""
        try:
            return self.load(name, device=device)

        except RuntimeError as e:
            # This error happens when there is a mismatch between save device and current device
            if 'CPU-only machine' in str(e):
                raise KeyboardInterrupt('Job got scheduled on bad node.') from e

        except FileNotFoundError:
            info(f'State file {name} was not found')
            return None
