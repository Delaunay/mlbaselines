from olympus.utils.gpu.nvidia import NvGpuMonitor
from olympus.utils.gpu.amd import AmdGpuMonitor

from enum import IntEnum
from multiprocessing import Process


class DeviceVendor(IntEnum):
    AMD = 0
    NVIDIA = 1


AMD = DeviceVendor.AMD
NVIDIA = DeviceVendor.NVIDIA


def get_device_vendor():
    import torch
    name: str = torch.cuda.get_device_name(0)

    if name.startswith('Ellesmere') or name.startswith('Vega'):
        return AMD

    return NVIDIA


def start_monitor(monitor):
    monitor.run()
    return monitor


def make_monitor(loop_interval=1000, device_id=0):
    vendor = get_device_vendor()

    if vendor is AMD:
        monitor = AmdGpuMonitor(loop_interval, device_id)
    else:
        monitor = NvGpuMonitor(loop_interval, device_id)

    proc = Process(target=start_monitor, args=(monitor,))
    proc.start()
    return proc, monitor


class GpuMonitor:
    def __init__(self, loop_interval=1000, device_id=0, enabled=True):
        self.loop = loop_interval
        self.device = device_id
        self.proc = None
        self.monitor = None
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            self.proc, self.monitor = make_monitor(self.loop, self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self.monitor.stop()
            self.proc.terminate()

    def to_json(self):
        if self.enabled:
            return self.monitor.to_json()
        return ['monitor disabled']

