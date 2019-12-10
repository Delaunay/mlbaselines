import torch
from torch.nn import Module

from olympus.utils import info, select
from olympus.tasks.task import Task
from olympus.metrics import OnlineLoss
from olympus.observers import ProgressView, Speed, ElapsedRealTime, CheckPointer, Tracker, SampleCount


class ObjectDetection(Task):
    def __init__(self, detector, optimizer, lr_scheduler, dataloader, criterion=None, device=None,
                 storage=None, logger=None):
        super(ObjectDetection, self).__init__(device=device)

        self._first_epoch = 0
        self.detector = detector
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dataloader = dataloader
        self.criterion = criterion
        self.storage = storage

        self.metrics.append(ElapsedRealTime().every(batch=1))
        self.metrics.append(SampleCount().every(batch=1, epoch=1))
        speed = Speed()
        self.metrics.append(speed)
        self.metrics.append(ProgressView(speed_observer=speed))
        self.metrics.append(OnlineLoss())

        if storage:
            self.metrics.append(CheckPointer(storage=storage))

        if logger is not None:
            self.metrics.append(Tracker(logger=logger))

    @property
    def model(self) -> Module:
        return self.detector

    @model.setter
    def model(self, model):
        self.detector = model

    def eval_loss(self, batch):
        # Will be fixed in the next-next torchvision release
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

    def fit(self, epochs, context=None):
        self._start(epochs)

        for epoch in range(self._first_epoch, epochs):
            self.epoch(epoch + 1, context)

        self.report(pprint=True, print_fun=print)
        self.metrics.finish(self)

    def epoch(self, epoch, context):
        for step, batch in enumerate(self.dataloader):
            self.step(step, batch, context)

        self.metrics.on_new_epoch(epoch, self, context)
        self.lr_scheduler.epoch(epoch, lambda x: self.metrics.value()['validation_loss'])

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

    def resume(self, storage):
        state_dict = storage.safe_load('checkpoint', device=self.device)

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

    def checkpoint(self, epoch, storage):
        info(f'Saving checkpoint (epoch: {epoch})')
        was_saved = storage.save(
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

        self.set_device(self.device)
        self.optimizer.init(
            self.detector.parameters(),
            override=True, **optimizer
        )
        self.lr_scheduler.init(
            self.optimizer,
            override=True, **lr_schedule
        )

        # try to resume itself
        checkpoints = self.metrics.get('CheckPointer')
        if checkpoints is not None:
            self.resume(checkpoints.storage)

        parameters = {}
        parameters.update(optimizer)
        parameters.update(lr_schedule)
        parameters.update(model)

        self.metrics.on_new_trial(self, parameters, trial_id)
        self.set_device(self.device)
