from olympus.utils import MissingArgument, warning, HyperParameters
from olympus.utils.factory import fetch_factories


registered_schedules = fetch_factories('olympus.optimizers.schedules', __file__)


class RegisteredLRSchedulerNotFound(Exception):
    pass


class UninitializedLRScheduler(Exception):
    pass


def known_schedule():
    return registered_schedules.keys()


def register_schedule(name, factory, override=False):
    global registered_schedules

    if name in registered_schedules:
        warning(f'{name} was already registered, use override=True to ignore')

        if not override:
            return

    registered_schedules[name] = factory


class LRSchedule:
    """Lazy LRSchedule that allows you to first fetch the supported parameters using ``get_space`` and then
    initialize the underlying schedule using ``init_optimizer``

    Parameters
    ----------
    name: str
       Name of a registered schedule

    schedule: LRSchedule
       Custom schedule, mutually exclusive with :param name

    Examples
    --------

    >>> schedule = LRSchedule('cyclic')
    >>> schedule.get_space()
    {'base_lr': 'loguniform(1e-5, 1e-2)', 'max_lr': 'loguniform(1e-2, 1)', ... }
    >>> schedule.init(optimizer, base_lr=1e-2, ...)

    Raises
    ------
    RegisteredLRSchedulerNotFound
        when using a name of an known schedule

    MissingArgument:
        if name nor schedule were not set
    """

    def __init__(self, name=None, *, schedule=None, optimizer=None, **kwargs):
        self._schedule = None
        self._schedule_builder = None
        self._optimizer = optimizer

        self.hyper_parameters = HyperParameters(space={})

        if schedule:
            if isinstance(schedule, type):
                self._schedule_builder = schedule

                if hasattr(schedule, 'get_space'):
                    self.hyper_parameters.space = schedule.get_space()

            else:
                self._schedule = schedule

            if hasattr(self._schedule, 'get_space'):
                self.hyper_parameters.space = self._schedule.get_space()

        elif name:
            # load an olympus model
            builder = registered_schedules.get(name)

            if not builder:
                raise RegisteredLRSchedulerNotFound(name)

            self._schedule_builder = builder

            if hasattr(self._schedule_builder, 'get_space'):
                self.hyper_parameters.space = self._schedule_builder.get_space()

        else:
            raise MissingArgument('None or name needs to be set')

        self.hyper_parameters.add_parameters(**kwargs)

    def init(self, optimizer=None, override=False, **kwargs):
        if self._schedule:
            warning('LRSchedule is already set, use override=True to force re initialization')

            if not override:
                return self._schedule

        if optimizer is None:
            optimizer = self._optimizer

        if optimizer is None:
            raise MissingArgument('Missing optimizer argument!')

        self._schedule = self._schedule_builder(optimizer, **kwargs)

        return self

    def get_space(self):
        if self._schedule:
            warning('LRSchedule is already set')

        return self.hyper_parameters.missing_parameters()

    def get_params(self, params):
        if self._schedule:
            warning('Optimizer is already set!')

        if self._schedule_builder:
            return self._schedule_builder.get_params(params)

        return {}

    @property
    def lr_scheduler(self):
        if not self._schedule:
            self.init()

        return self._schedule

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict)

    def epoch(self, epoch, metrics=None):
        self.lr_scheduler.epoch(epoch, metrics)

    def step(self, step, metrics=None):
        self.lr_scheduler.step(step, metrics)

    def get_lr(self):
        return self.lr_scheduler.get_lr()
