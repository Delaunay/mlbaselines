from datetime import datetime
import os
import io
from shutil import copyfile
import tempfile
import torch
import json

from filelock import FileLock

from olympus.utils import info
from olympus.utils.signals import Protected
from olympus.utils.options import options


class BaseStorage:
    Kio = 1024
    Mio = 1024 * 1024

    def load(self, *args, **kwargs):
        pass

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

    def __init__(self, *args, **kwargs):
        pass

    def set_base(self, *args, **kwargs):
        pass

    def save_meta(self, uid, meta):
        pass

    def show_memory_usage(self):
        return {}

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

    def rename(self, old, new):
        pass

    def copyfile(self, old, new):
        pass

    def remove(self, file):
        pass


def NoStorage(*args, **kwargs):
    return BaseStorage(*args, **kwargs)


def safe_write(filename, buffer):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    fd, name = tempfile.mkstemp('meta', dir=dirname)

    file = os.fdopen(fd, 'wb')
    file.write(buffer)
    file.close()

    os.rename(name, filename)


class InMemoryMetaStorage:
    """Provide reverse mapping from uid back to parameters"""
    def __init__(self):
        self.cache = dict()

    def save(self, folder, uid, meta):
        self.cache[uid] = meta

    def load(self, folder, uid=None):
        data = self.cache

        if uid is not None:
            return data.get(uid, dict())

        return data


class FileMetaStorage:
    """Provide reverse mapping from uid back to parameters"""
    def __init__(self, folder):
        self.lock = FileLock(os.path.join(folder, 'meta.lock'), timeout=3)

    def loc(self, folder):
        return os.path.join(folder, 'meta.json')

    def save(self, folder, uid, meta):
        filename = self.loc(folder)
        with self.lock:
            if os.path.exists(filename):
                with open(filename, 'r') as fp:
                    data = json.load(fp)
            else:
                data = dict()

            old_meta = data.get(uid, dict())
            old_meta.update(meta)
            data[uid] = old_meta

            safe_write(filename, json.dumps(data).encode('utf-8'))

    def load(self, folder, uid=None):
        with open(self.loc(folder), 'r') as fp:
            data = json.load(fp)

        if uid is not None:
            return data.get(uid, dict())

        return data


class InMemoryStorage(BaseStorage):
    """Save states in memory

    Parameters
    ----------
    format: str
        Which format is used to save the state, default to dict (i.e native python state dict)
        It can also be set to bytes to have a format that is writable directly to disc
    """
    def __init__(self, format='dict'):
        self.format = format
        self.cache = dict()
        self.meta = InMemoryMetaStorage()
        self.in_memory = 0

    def write(self, filename, data):
        self.cache[filename] = data

    def read(self, filename):
        return self.cache.get(filename)

    def exits(self, filename):
        return filename in self.cache

    def rename(self, old, new):
        state = self.pop_from_cache(old)
        self.save(new, state)

    def load_meta(self):
        self.meta.load(None, None)

    def save_meta(self, uid, meta):
        self.meta.save(None, uid, meta)

    def remove(self, file):
        self.cache.pop(file, None)

    def save(self, filename, state):
        buffer = state

        if self.format == 'bytes':
            # Writes the state inside a buffer
            buffer = io.BytesIO()
            torch.save(state, buffer)
            buffer = buffer.getbuffer()

        self.insert_cache(filename, buffer)
        return True

    def load(self, filename, device=None):
        """

        Parameters
        ----------
        filename: str
            file to load the state from

        device: torch.device
            it indicates the location where all tensors should be loaded.
        """
        buffer = self.read(filename)

        if self.format == 'bytes':
            return torch.load(buffer, map_location=lambda storage, loc: storage)

        return buffer

    def show_memory_usage(self):
        return {
            'in_memory': self.in_memory / BaseStorage.Mio,
            'count': len(self.cache)
        }

    def garbage_collect(self, gc_time):
        now = datetime.utcnow()
        old = self.in_memory
        to_be_deleted = []

        for name, (buffer, save_time) in self.cache.items():
            if (now - save_time).total_seconds > gc_time:
                to_be_deleted.append(name)

        for name in to_be_deleted:
            self.pop_from_cache(name)

        new = self.in_memory
        freed = old - new
        return freed

    def insert_cache(self, key, buffer):
        if key in self.cache:
            self.pop_from_cache(key)

        self.cache[key] = (buffer, datetime.utcnow())
        if self.format == 'bytes':
            self.in_memory += buffer.getbuffer().nbytes

    def pop_from_cache(self, key) -> bytes:
        buffer, _ = self.cache.pop(key, (None, None))

        if buffer and self.format == 'bytes':
            self.in_memory -= buffer.getbuffer().nbytes

        return buffer


class FileStateStorage(BaseStorage):
    def __init__(self, folder=options('state.storage', '/tmp')):
        # typically root/task_name/experiment_name/trial_id
        self.folder = None
        self.set_base(folder)
        self.last_save = datetime.utcnow()

        self.on_disk = 0
        self.on_disk_files = dict()
        self.meta = FileMetaStorage(self.folder)

    def load_meta(self):
        self.meta.load(self.folder, None)

    def set_base(self, folder):
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)

    def show_memory_usage(self):
        return {
            'on_disk': self.on_disk / BaseStorage.Mio,
            'on_disk_file_count': len(self.on_disk_files)
        }

    def garbage_collect(self, gc_time):
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

    def _file(self, filename):
        return f'{self.folder}/{filename}.state'

    def rename(self, old, new):
        os.rename(self._file(old), self._file(new))

    def copyfile(self, old, new):
        copyfile(self._file(old), self._file(new))

    def open(self, filename, mode):
        return open(self._file(filename), mode)

    def write(self, filename, data):
        return self.open(filename, 'w').write(data)

    def read(self, filename):
        return self.open(filename, 'r').read()

    def exits(self, filename):
        return os.path.exists(self._file(filename))

    def save_meta(self, uid, meta):
        self.meta.save(self.folder, uid, meta)

    def remove(self, file):
        try:
            os.remove(self._file(file))
        except FileNotFoundError:
            pass

    def save(self, filename, state):
        with Protected():
            return self._save(filename, state)

    def _save(self, filename, state):
        from olympus.utils import info
        path = self._file(filename)

        # Writes the state inside a buffer
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer = buffer.getbuffer()

        safe_write(path, buffer)

        # Remove from cache it is in
        self._insert_disk(filename, buffer.nbytes)
        self.last_save = datetime.utcnow()
        return True

    def _insert_disk(self, key, size):
        if key in self.on_disk_files:
            self._pop_from_disk(key)

        self.on_disk_files[key] = (size, datetime.utcnow())
        self.on_disk += size

    def _pop_from_disk(self, key):
        size, _ = self.on_disk_files.pop(key, (None, None))
        if size:
            self.on_disk -= size

    def load(self, filename, device=None):
        """

        Parameters
        ----------
        filename: str
            file to load the state from

        device: torch.device
            it indicates the location where all tensors should be loaded.
        """
        buffer = self._file(filename)
        return torch.load(buffer, map_location=lambda storage, loc: storage)


StateStorage = FileStateStorage
