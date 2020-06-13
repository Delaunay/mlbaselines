import torch

from olympus.tasks.task import Task
from olympus.metrics import OnlineLoss
from olympus.metrics.named import NamedMetric
from olympus.resuming import BadResumeGuard, load_state_dict, state_dict
from olympus.observers import ProgressView, Speed, ElapsedRealTime, SampleCount
from olympus.utils import select, drop_empty_key


class SharpeRatio:
    """
    https://en.wikipedia.org/wiki/Sharpe_ratio

    You want to maximize the Sharpe ratio

    Examples
    --------
    >>> batch_size = 3
    >>> cov = torch.randn(batch_size, 10, 10)
    >>> returns = torch.randn(batch_size, 10)
    >>> weight = torch.randn(batch_size, 10)
    >>> crit = SharpeRatio()
    >>> sharp_ratios = crit(weight, cov, returns)
    >>> sharp_ratios.shape
    torch.Size([3, 1])
    """
    def __init__(self):
        self.std = None
        self.returns = None

    def __call__(self, weight, cov, returns):
        if len(weight.shape) != 3:
            weight = weight.unsqueeze(2)

        if len(returns.shape) != 3:
            returns = returns.unsqueeze(2)

        self.returns = torch.matmul(returns.transpose(2, 1), weight)
        self.std = torch.sqrt(torch.matmul(torch.matmul(weight.transpose(2, 1), cov), weight))

        return (self.returns / self.std).squeeze(2)


class SharpeRatioCriterion:
    def __init__(self):
        self.ratio = SharpeRatio()

    def __call__(self, w, c, r):
        return - self.ratio(w, c, r).mean()

    def mean_returns(self):
        return self.ratio.returns.mean().detach()

    def std_returns(self):
        return self.ratio.std.mean().detach()


class Finance(Task):
    def __init__(self, dataset, oracle, model, optimizer, device, criterion=SharpeRatioCriterion()):
        super(Finance, self).__init__(device=device)

        self.dataset = dataset
        self.oracle = oracle
        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer

        self.metrics.append(ElapsedRealTime().every(batch=1))
        self.metrics.append(SampleCount().every(batch=1, epoch=1))
        speed = Speed()
        self.metrics.append(speed)
        self.metrics.append(ProgressView(speed))
        self.metrics.append(OnlineLoss())
        self.metrics.append(NamedMetric(name='mean_returns'))
        self.metrics.append(NamedMetric(name='std_returns'))
        self.current_epoch = 0
        self.hyper_parameters = {}

    def fit(self, epochs, context=None):
        if self.stopped:
            return

        with BadResumeGuard(self):
            self._start(epochs)

            for epoch in range(self._first_epoch, epochs):
                self._fix()
                self.current_epoch = epoch + 1
                self.metrics.new_epoch(epoch + 1, context)

                for step, s in enumerate(self.dataset):
                    self.metrics.new_batch(step, s)

                    results = self.step(step, s, context)

                    self.metrics.end_batch(step, s, results)

                self.metrics.end_epoch(epoch + 1, context)

                if self.stopped:
                    break

            self.report(pprint=True, print_fun=print)
            self.metrics.end_train()

    def step(self, step, input, context):
        x_train, x_target = input
        x_train, x_target = x_train[0].cuda(), x_target.cuda()

        weight_predict = self.model(x_train).cuda()

        with torch.no_grad():
            cov, ret = self.oracle(x_target)

        loss = self.criterion(weight_predict, cov, ret)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return dict(
            loss=loss.detach(),
            mean_returns=self.criterion.mean_returns(),
            std_returns=self.criterion.std_returns())

    def eval_loss(self, batch):
        self.model.train()

        with torch.no_grad():
            x_train, x_target = batch
            x_train, x_target = x_train[0].to(device=self.device), x_target.to(device=self.device)

            weight_predict = self.model(x_train)
            cov, ret = self.oracle(x_target)
            loss = self.criterion(weight_predict.to(device=self.device), cov, ret)

        self.model.train()

        # do not use item() in the loop it forces cuda to sync
        if hasattr(loss, 'detach'):
            return loss.detach()

        return torch.Tensor(loss)

    def load_state_dict(self, state, strict=True):
        load_state_dict(self, state, strict, force_default=True)
        self._first_epoch = state['epoch']
        self.current_epoch = state['epoch']

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = state_dict(self, destination, prefix, keep_vars, force_default=True)
        state['epoch'] = self.current_epoch
        return state

    def get_space(self):
        """Return hyper parameter space"""
        return drop_empty_key({
            'optimizer': self.optimizer.get_space(),
            'model': self.model.get_space()
        })

    def get_current_space(self):
        """Get currently defined parameter space"""
        return {
            'optimizer': self.optimizer.get_current_space(),
            'model': self.model.get_current_space()
        }

    def init(self, optimizer=None, model=None, uid=None):
        """
        Parameters
        ----------
        optimizer: Dict
            Optimizer hyper parameters!s

        model: Dict
            model hyper parameters

        uid: Optional[str]
            trial id to use for logging.
            When using orion usually it already created a trial for us we just need to append to it
        """
        optimizer = select(optimizer, {})
        model = select(model, {})

        self.model.init(**model)

        # list of all parameters this task has
        parameters = list()
        parameters.append({
            'params': self.model.parameters()}
        )

        # We need to set the device now so optimizer receive cuda tensors
        self.set_device(self.device)
        self.optimizer.init(self.model.parameters(), **optimizer)

        self.hyper_parameters = {
            'optimizer': optimizer,
            'model': model
        }

        # Get all hyper parameters even the one that were set manually
        hyperparameters = self.get_current_space()

        # Trial Creation and Trial resume
        self.metrics.new_trial(hyperparameters, uid)
        self.set_device(self.device)
