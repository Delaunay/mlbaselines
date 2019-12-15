import torch
from torch.nn import Module

from olympus.utils import info, select
from olympus.tasks.task import Task
from olympus.metrics import OnlineLoss
from olympus.resuming import state_dict, load_state_dict, BadResumeGuard
from olympus.observers import ProgressView, Speed, ElapsedRealTime, CheckPointer, Tracker, SampleCount


class ObjectDetection(Task):
    def __init__(self, detector, optimizer, lr_scheduler, dataloader, criterion=None, device=None,
                 storage=None, logger=None):
        super(ObjectDetection, self).__init__(device=device)

        self._first_epoch = 0
        self.current_epoch = 0
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

    # Hyper Parameter Settings
    # ---------------------------------------------------------------------
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

    def init(self, optimizer=None, lr_schedule=None, model=None, trial=None):
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

        parameters = {}
        parameters.update(optimizer)
        parameters.update(lr_schedule)
        parameters.update(model)

        # Trial Creation and Trial resume
        self.metrics.on_new_trial(self, parameters, trial)
        self.set_device(self.device)

    # Training
    # --------------------------------------------------------------------
    def fit(self, epochs, context=None):
        with BadResumeGuard(self):
            self._start(epochs)

            for epoch in range(self._first_epoch, epochs):
                self.epoch(epoch + 1, context)

            self.report(pprint=True, print_fun=print)
            self.metrics.finish(self)

    def epoch(self, epoch, context):
        self.current_epoch = epoch

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
    # ---------------------------------------------------------------------

    def load_state_dict(self, state, strict=True):
        load_state_dict(self, state, strict, force_default=True)
        self._first_epoch = state['epoch']
        self.current_epoch = state['epoch']

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = state_dict(self, destination, prefix, keep_vars, force_default=True)
        state['epoch'] = self.current_epoch
        return state

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

    @property
    def model(self) -> Module:
        return self.detector

    @model.setter
    def model(self, model):
        self.detector = model

    def parameters(self):
        return self.detector.parameters()
