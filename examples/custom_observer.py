from dataclass import dataclass
from olympus.observer import Observer


@dataclass
class ProgressPrinter(Observer):
    frequency_epoch: int = 1    # run every epoch
    frequency_batch: int = 100  # run every 100 batch

    def on_new_batch(self, step, task=None, input=None, context=None):
        print('step', step)

    def on_new_epoch(self, epoch, task=None, context=None):
        print('epoch', epoch)

