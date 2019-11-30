import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from olympus.utils import info, select
from olympus.tasks.task import Task, BadResumeGuard
from olympus.optimizers.schedules import LRSchedule
from olympus.utils.storage import StateStorage, NoStorage
from olympus.metrics import OnlineLoss
from olympus.observers import ElapsedRealTime, SampleCount, ProgressView


class ObjectDetection(Task):
    detector: Module
    optimizer: Optimizer
    criterion: Module
    dataloader: DataLoader
    lr_scheduler: LRSchedule
    storage: StateStorage = NoStorage()
    _first_epoch: int = 0

    def __init__(self, detector, optimizer, lr_scheduler, dataloader, criterion=None, device=None,
                 storage=None, logger=None):
        super(ObjectDetection, self).__init__(device=device, logger=logger)

        self._first_epoch = 0
        self.detector = detector
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dataloader = dataloader
        self.criterion = criterion
        self.storage = storage

        self.metrics.append(ElapsedRealTime().every(batch=1))
        self.metrics.append(SampleCount().every(batch=1, epoch=1))
        self.metrics.append(ProgressView())
        self.metrics.append(OnlineLoss())

    @property
    def model(self) -> Module:
        return self.detector

    @model.setter
    def model(self, model):
        self.detector = model

    def eval_loss(self, batch):
        self.model.train()

        with torch.no_grad():
            images, targets = batch

            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            loss = self.criterion(loss_dict)

        self.model.train()

        # do not use item() in the loop it forces cuda to sync
        if hasattr(loss, 'detach'):
            return loss.detach()

        return torch.Tensor(loss)

    def resumed(self):
        return self._first_epoch > 0

    def fit(self, epochs, context=None):
        with BadResumeGuard(self):
            self.detector.to(self.device)
            progress = self.metrics.get('ProgressView')

            if progress:
                # in case of a resume
                progress.epoch = self._first_epoch
                progress.max_epoch = epochs
                progress.max_step = len(self.dataloader)

            with self.logger as trial_logger:
                if not self.resumed():
                    self.metrics.start(self)
                    self.report(pprint=True, print_fun=print)
                    trial_logger.log_metrics(step=0, **self.metrics.value())

                for epoch in range(self._first_epoch, epochs):
                    self.epoch(epoch + 1, context)
                    self.report(pprint=True, print_fun=print)

                    if epoch != epochs - 1:
                        trial_logger.log_metrics(step=epoch + 1, **self.metrics.value())

                self.metrics.finish(self)
                trial_logger.log_metrics(step=epochs, **self.metrics.value())

    def epoch(self, epoch, context):
        for step, batch in enumerate(self.dataloader):
            self.step(step, batch, context)

        self.metrics.on_new_epoch(epoch, self, context)
        self.lr_scheduler.epoch(epoch, lambda x: self.metrics.value()['validation_loss'])
        self.checkpoint(epoch)

    def step(self, step, input, context):
        images, targets = input

        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        self.model.train()
        self.optimizer.zero_grad()

        loss_dict = self.model(images, targets)
        loss = self.criterion(loss_dict)
        loss.backward()

        self.optimizer.step()

        results = {
            # to compute online loss
            'loss': loss.detach()
        }

        self.metrics.on_new_batch(step, self, input, results)
        self.lr_scheduler.step(step)
        return results

    def resume(self):
        with BadResumeGuard(self):
            state_dict = self.storage.safe_load('checkpoint', device=self.device)

            if not state_dict:
                info('Starting from scratch')
                return False

            try:
                self._first_epoch = state_dict['epoch']
                info(f"Resuming from (epoch: {self._first_epoch})")

                self.model.load_state_dict(state_dict['model'])
                self.optimizer.load_state_dict(state_dict['optimizer'], device=self.device)
                self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
                self.dataloader.sampler.load_state_dict(state_dict['sampler'])
                self.metrics.load_state_dict(state_dict['metrics'])

                return True
            except KeyError as e:
                raise KeyError(f'Bad state dictionary!, missing (key: {e.args})') from e

    def checkpoint(self, epoch):
        info(f'Saving checkpoint (epoch: {epoch})')
        was_saved = self.storage.save(
            'checkpoint',
            dict(
                epoch=epoch,
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                lr_scheduler=self.lr_scheduler.state_dict(),
                sampler=self.dataloader.sampler.state_dict(),
                metrics=self.metrics.state_dict()
            )
        )
        if was_saved:
            info('Checkpoint saved')
        else:
            info('Skipped Checkpoint')

    def get_space(self, **fidelities):
        """Return hyper parameter space"""
        return {
            'task': {       # fidelity(min, max, base logarithm)
                'epochs': fidelities.get('epochs')
            },
            'optimizer': self.optimizer.get_space(),
            'lr_schedule': self.lr_scheduler.get_space(),
            'model': self.model.get_space()
        }

    def init(self, optimizer=None, lr_schedule=None, model=None, trial_id=None):
        optimizer = select(optimizer, {})
        lr_schedule = select(lr_schedule, {})
        model = select(model, {})

        self.detector.init(
            **model
        )
        self.optimizer.init(
            self.detector.parameters(),
            override=True, **optimizer
        )
        self.lr_scheduler.init(
            self.optimizer,
            override=True, **lr_schedule
        )

        # try to resume itself
        self.resume()

        parameters = {}
        parameters.update(optimizer)
        parameters.update(lr_schedule)
        parameters.update(model)

        self.logger.upsert_trial(parameters, trial_id=trial_id)
        self.set_device(self.device)
