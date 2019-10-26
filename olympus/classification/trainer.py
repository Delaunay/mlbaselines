import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler

from collections import namedtuple

from olympus.utils.stat import StatStream
from olympus.utils.trainer import Trainer
from olympus.utils.sampler import ResumableSampler
from olympus.utils.optimizer import OptimizerAdapter
from olympus.utils.dtypes import Tensor, N, NCHW
from olympus.utils.log import TrainLogger
from olympus.utils import warning


class TrainClassifier(Trainer):
    # FIXME: I don't like how resumable sampler is handled

    def __init__(self,
                 optimizer: optim.Optimizer,
                 criterion: nn.Module,
                 model: nn.Module,
                 sampler: Sampler = None,
                 device=None):
        super(TrainClassifier, self).__init__()

        if not issubclass(type(optimizer), OptimizerAdapter):
            optimizer = OptimizerAdapter(optimizer)

        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model
        self.logger = TrainLogger()

        self.epoch_time = StatStream(drop_first_obs=10)
        self.batch_time = StatStream(drop_first_obs=10)
        self.sampler = sampler
        self.batch_count = 0

        self.device = device
        if device is None:
            self.device = torch.device('cpu')

        self.model = self.model.to(device)

    def fit(self, epochs: int, dataloader: DataLoader, *args, **kwargs):
        train_start = time.time()
        data_set_size = len(dataloader)
        epoch_time = self.epoch_time
        batch_time = self.batch_time
        self.model.train()

        self.logger.batch_count = data_set_size
        self.logger.epoch_count = epochs

        for epoch in range(epochs):
            epoch_start = time.time()

            loss = 0
            loading_start = time.time()

            for batch_id, batch in enumerate(dataloader):
                self.batch_count += 1

                loading_end = time.time()
                loading_time = loading_end - loading_start

                loss += TrainClassifier.step(**locals())
                loading_start = time.time()

            epoch_end = time.time()
            epoch_time.update(epoch_end - epoch_start)

            loss /= data_set_size
            self.logger.log_metric('train_loss', loss)

            # ---
            v = locals()
            v.pop('self')
            self.logger.log_epoch(**v)
            # ---

        train_end = time.time()
        train_time = train_end - train_start

        # ---
        v = locals()
        v.pop('self')
        self.logger.log_train(**v)
        # ---

    def step(self, batch: (Tensor[NCHW], Tensor[N]), loading_time, batch_time, **kwargs):
        compute_start = time.time()
        self.model.train()
        batch_loss = self.batch(batch)

        compute_end = time.time()
        compute_time = compute_end - compute_start

        batch_time.update(loading_time + compute_time)

        # ---
        scope = self._merge_env(locals(), kwargs)
        scope.pop('self')
        self.logger.log_step(**scope)
        # ---

        return batch_loss

    def batch(self, batch: (Tensor[NCHW], Tensor[N])):
        data, label = batch
        data = data.to(self.device)
        label = label.to(self.device)

        self.optimizer.zero_grad()

        p_label = self.model(data)
        oloss = self.criterion(p_label, label)
        batch_loss = oloss.item()

        # compute gradient and do SGD step
        self.optimizer.backward(oloss)
        self.optimizer.step()
        return batch_loss

    def eval_model(self, dataloader: DataLoader):
        """Evaluate performance of a model for a given set of data"""
        self.model.eval()
        loss = 0
        correct = 0
        size = len(dataloader.dataset)

        with torch.no_grad():
            for batch_id, (data, target) in enumerate(dataloader):
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)

                loss += self.criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        loss /= size
        acc = correct / size
        EvalTuple = namedtuple('EvalPoint', ['acc', 'loss'])
        return EvalTuple(acc, loss)

    def infer(self, batch):
        self.model.eval()

        with torch.no_grad():
            output = self.model(batch)
            return output.argmax(dim=1, keepdim=True)

    @staticmethod
    def _merge_env(scope, env):
        for k, v in env.items():
            scope[k] = v
        return scope

    def resume(self, path, find_latest=True):
        import glob

        if find_latest:
            stuff = path.split('.')
            path = '.'.join(stuff[:-1])

            files = list(glob.glob(f'{path}*'))
            files.sort()

            path = files[-1]

        state = torch.load(path)
        self.optimizer.load_state_dict(state['optimizer'])
        self.model.load_state_dict(state['model'])
        self.batch_count = state['batch_count']
        self.epoch_time = StatStream.from_dict(state['epoch_time'])
        self.batch_time = StatStream.from_dict(state['batch_time'])
        self.sampler = ResumableSampler(self.sampler, state=state['sampler'].get('rng_state'))
        torch.set_rng_state(state['rng_state'])

    def save(self, path, override=True):
        import glob
        import os

        stuff = path.split('.')
        ext = stuff[-1]
        path = '.'.join(stuff[:-1])

        if not override:
            n = len(glob.glob(f'{path}*'))
            if n > 0:
                path = f'{path}_{self.batch_count}_{n}'

        path = f'{path}.{ext}'
        temp_path = f'{path}.temp'

        torch.save({
            'batch_count': self.batch_count,
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.state_dict(),
            'epoch_time': self.epoch_time.state_dict(),
            'batch_time': self.batch_time.state_dict(),
            'sampler': self.sampler.state_dict(),
            'rng_state': torch.get_rng_state()
        }, temp_path)
        os.rename(temp_path, path)
