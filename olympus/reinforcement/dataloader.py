import torch

from olympus.utils.dataloader import DataLoader, GenericStateIterator
from olympus.reinforcement.environment import ParallelEnvironment


class _RLIterator(GenericStateIterator):
    """Make gym environment behave like iterators

    Parameters
    ----------
    env: Env
        Gym environment or ParallelEnvironment from mlbaseline

    max_steps: int
        Max Frame / steps to generate. Once the max step is reached the iterator returns None
    """

    def __init__(self, loader, env, max_steps: int = None):
        super(_RLIterator, self).__init__(loader)

        self.env = env
        self.max_steps = max_steps
        self.init = False
        self.loader = loader

    def next(self, action):
        if self.init is False:
            assert action is None
            self.init = True

            self.loader.batch_id += 1
            return self.env.reset()

        if self.max_steps is not None and self.loader.batch_id > self.max_steps:
            self.loader.epoch_id += 1
            self.loader.batch_id = 0
            return None

        self.loader.batch_id += 1
        return self.env.step(action)


class RLDataloader(DataLoader):
    """
    Parameters
    ----------
    num_workers: int
        number of simulation/game/environment running in parallel

    max_steps: int
        stop the simulation after max_steps steps

    env_factory: Callable[]
        Pickable function to be called to initialize the environment on all workers

    env_args:
        args to pass to the env_factory

    Notes
    -----
    Depending on the simulation num_workers can be greatly superior to the number of threads.
    Users should monitor GPU usage and adjust accordingly

    """
    def __init__(self, num_workers: int, max_steps: int, state_transforms, env_factory, *env_args):
        super(RLDataloader, self).__init__()

        if num_workers > 1:
            self.env = ParallelEnvironment(num_workers, state_transforms, env_factory, *env_args)
        else:
            # FIXME
            print('num_workers=1 is broken')
            self.env = env_factory(*env_args)

        self._batch_shape = None
        self.batch_size = num_workers
        self.max_steps = max_steps

    def iterator(self):
        return _RLIterator(self, self.env, self.max_steps)

    @property
    def batch_shape(self):
        if self._batch_shape is None:
            # Transform might modify the shape of the state so we have to compute the shape using a dummy state
            with torch.no_grad():
                tracer = torch.randn((1,) + self.env.observation_space.shape)

                if hasattr(self.env, 'transforms'):
                    tracer = self.env.transforms(tracer)
                self._batch_shape = (self.batch_size,) + tracer.shape[1:]

        return self._batch_shape

    @property
    def state_vector_shape(self):
        return self.batch_shape[1:]

    @property
    def action_vector_size(self):
        return self.env.action_space.n
