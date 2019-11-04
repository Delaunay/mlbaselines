from olympus.utils import MissingArgument, warning
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
    >>> schedule.init_schedule(optimizer, base_lr=1e-2, ...)

    Raises
    ------
    RegisteredLRSchedulerNotFound
        when using a name of an known schedule

    MissingArgument:
        if name nor schedule were not set
    """

    def __init__(self, name=None, schedule=None):
        self._schedule = None

        if schedule:
            self._schedule = schedule

        elif name:
            # load an olympus model
            self._schedule_builder = registered_schedules.get(name)()

            if not self._schedule_builder:
                raise RegisteredLRSchedulerNotFound(name)

        else:
            raise MissingArgument('None or name needs to be set')

    def init_schedule(self, optimizer, override=False, **kwargs):
        if self._schedule:
            warning('LRSchedule is already set, use override=True to force re initialization')

            if not override:
                return self._schedule

        self._schedule = self._schedule_builder(optimizer, **kwargs)

        return self

    def get_space(self):
        if self._schedule:
            warning('LRSchedule is already set')

        if self._schedule_builder:
            return self._schedule_builder.get_space()

        return {}

    def get_params(self, params):
        if self._schedule:
            warning('Optimizer is already set!')

        if self._schedule_builder:
            return self._schedule_builder.get_params(params)

        return {}

    @property
    def lr_scheduler(self):
        if not self._schedule:
            raise UninitializedLRScheduler('Call `init_schedule` first')

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
