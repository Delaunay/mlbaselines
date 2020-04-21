import torch

from torch.nn import Module, CrossEntropyLoss

from olympus.observers import ElapsedRealTime, SampleCount, ProgressView, Speed, CheckPointer
from olympus.metrics import OnlineTrainAccuracy
from olympus.tasks.task import Task
from olympus.utils import select, drop_empty_key
from olympus.resuming import state_dict, load_state_dict, BadResumeGuard
from olympus.transforms import Preprocessor


class Classification(Task):
    """Train a model to recognize a range of classes

    Attributes
    ----------
    classifier: Module
        Module taking sample data and returning the probability of the sample belonging to a range of classes

    optimizer: Optimizer
        Optimizer taking model's parameters

    criterion: Module
        Function evaluating the quality of the model's predictions, also named cost function or loss function

    lr_scheduler: LRSchedule
        Learning Scheduler, updates the learning rates periodically

    dataloader: Iterator
        Batch sample iterator used to train the model

    preprocessor: Preprocessor
        Set of functions that transform the inputs before it is given to the model

    device:
        Acceleration device to run the task on

    storage: Storage
        Where to save checkpoints in case of failures
    """
    def __init__(self, classifier, optimizer, lr_scheduler, dataloader, criterion=None, device=None,
                 storage=None, preprocessor=None):
        super(Classification, self).__init__(device=device)
        criterion = select(criterion, CrossEntropyLoss())

        self._first_epoch = 0
        self.current_epoch = 0
        self.classifier = classifier
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dataloader = dataloader
        self.criterion = criterion
        self.preprocessor = Preprocessor()
        # ------------------------------------------------------------------
        # TODO: This should go inside user code it will remove 2 arguments
        self.metrics.append(ElapsedRealTime().every(batch=1))
        self.metrics.append(SampleCount().every(batch=1, epoch=1))
        self.metrics.append(OnlineTrainAccuracy())
        self.metrics.append(Speed())
        self.metrics.append(ProgressView(speed_observer=self.metrics.get('Speed')))

        if storage:
            self.metrics.append(CheckPointer(storage=storage))
        # ------------------------------------------------------------------

        if preprocessor is not None:
            self.preprocessor = preprocessor

        self.hyper_parameters = {}

    # Hyper Parameter Settings
    # ---------------------------------------------------------------------
    def get_space(self):
        """Return hyper parameter space"""
        return drop_empty_key({
            'optimizer': self.optimizer.get_space(),
            'lr_schedule': self.lr_scheduler.get_space(),
            'model': self.model.get_space()
        })

    def get_current_space(self):
        """Get currently defined parameter space"""
        return {
            'optimizer': self.optimizer.get_current_space(),
            'lr_schedule': self.lr_scheduler.get_current_space(),
            'model': self.model.get_current_space()
        }

    def init(self, optimizer=None, lr_schedule=None, model=None, uid=None):
        """
        Parameters
        ----------
        optimizer: Dict
            Optimizer hyper parameters!s

        lr_shchedule: Dict
            lr schedule hyper parameters

        model: Dict
            model hyper parameters

        trial_id: Optional[str]
            trial id to use for logging.
            When using orion usually it already created a trial for us we just need to append to it
        """
        optimizer = select(optimizer, {})
        lr_schedule = select(lr_schedule, {})
        model = select(model, {})

        self.classifier.init(
            **model
        )

        # list of all parameters this task has
        parameters = self.preprocessor.parameters()
        parameters.append({
            'params': self.classifier.parameters()}
        )

        # We need to set the device now so optimizer receive cuda tensors
        self.set_device(self.device)
        self.optimizer.init(
            parameters,
            override=True, **optimizer
        )
        self.lr_scheduler.init(
            self.optimizer,
            override=True, **lr_schedule
        )

        self.hyper_parameters = {
            'optimizer': optimizer,
            'lr_schedule': lr_schedule,
            'model': model
        }

        # Get all hyper parameters even the one that were set manually
        hyperparameters = self.get_current_space()

        # Trial Creation and Trial resume
        self.metrics.new_trial(hyperparameters, uid)
        self.set_device(self.device)

    # Training
    # ---------------------------------------------------------------------
    def fit(self, epochs, context=None):
        with BadResumeGuard(self):
            self.classifier.to(self.device)
            self._start(epochs)

            for epoch in range(self._first_epoch, epochs):
                self.epoch(epoch + 1, context)

            self.metrics.end_train()
            self._first_epoch = epochs

    def epoch(self, epoch, context):
        self.current_epoch = epoch
        self.metrics.new_epoch(epoch, context)

        for step, mini_batch in enumerate(self.dataloader):
            self.metrics.new_batch(step, mini_batch, None)

            results = self.step(step, mini_batch, context)

            self.lr_scheduler.step(step)
            self.metrics.end_batch(step, mini_batch, results)

        self.lr_scheduler.epoch(epoch, self._get_validation_accuracy)
        self.metrics.end_epoch(epoch, context)

    def step(self, step, input, context):
        self.classifier.train()
        self.optimizer.zero_grad()

        preprocessed_input = self.preprocessor(input)

        batch = preprocessed_input['batch']
        target = preprocessed_input['target']

        batch = [x.to(device=self.device) for x in batch]
        predictions = self.classifier(*batch)
        loss = self.criterion(predictions, target.to(device=self.device))

        self.optimizer.backward(loss)
        self.optimizer.step()

        results = {
            # to compute online loss
            'loss': loss.detach(),
            # to compute only accuracy
            'predictions': predictions.detach()
        }

        return results
    # ---------------------------------------------------------------------

    def _get_validation_accuracy(self, x):
        return self.metrics.value().get('validation_accuracy', None)

    def eval_loss(self, batch):
        self.model.eval()

        with torch.no_grad():
            batch, target = batch
            batch = [x.to(device=self.device) for x in batch]
            predictions = self.classifier(*batch)
            loss = self.criterion(predictions, target.to(device=self.device))

        self.model.train()
        return loss.detach()

    def predict_probabilities(self, batch):
        with torch.no_grad():
            data = [x.to(device=self.device) for x in batch]
            return self.classifier(*data)
            return self.classifier(batch.to(device=self.device))

    def predict(self, batch, target=None):
        probabilities = self.predict_probabilities(batch)
        _, predicted = torch.max(probabilities, 1)

        loss = None
        if target is not None:
            loss = self.criterion(probabilities, target.to(device=self.device))

        return predicted, loss

    def accuracy(self, batch, target):
        predicted, loss = self.predict(batch, target)
        acc = (predicted == target.to(device=self.device)).sum()

        return acc.float() / target.size(0), loss

    def load_state_dict(self, state, strict=True):
        load_state_dict(self, state, strict, force_default=True)
        self._first_epoch = state['epoch']
        self.current_epoch = state['epoch']

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = state_dict(self, destination, prefix, keep_vars, force_default=True)
        state['epoch'] = self.current_epoch
        return state

    def parameters(self):
        return self.classifier.parameters()

    @property
    def model(self) -> Module:
        return self.classifier

    @model.setter
    def model(self, model):
        self.classifier = model
