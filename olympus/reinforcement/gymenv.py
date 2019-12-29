import functools

import numpy as np
import gym
from gym.envs.registration import registry as env_registry

from olympus.reinforcement.parallel import ParallelEnvironment


# TODO: figure out seeding
class GymEnvironment:
    def __init__(self, env_name, transforms=None, rand_seed=None, train_size=1024, valid_size=128, test_size=128, parallel_env=8,
                 num_thread=4,
                 distribution_mode='easy'):
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.env_name = env_name
        self.parallel_env = parallel_env
        self.num_thread = num_thread
        self.distribution_mode = distribution_mode
        self.seed = rand_seed
        self.transforms = transforms

        self._train = None
        self._valid = None
        self._test = None

    def _start_train(self):
        return 0

    def _start_valid(self):
        return self._start_train() + self.train_size

    def _start_test(self):
        return self._start_train() + self.train_size + self.valid_size

    def close(self):
        self._train.close()
        self._valid.close()
        self._test.close()

        self._train = None
        self._valid = None
        self._test = None

    @property
    def state_space(self):
        return self.input_size

    @property
    def action_space(self):
        return self.target_size

    @property
    def input_size(self):
        """Return the size of the samples"""
        return self.train.observation_space

    @property
    def target_size(self):
        """Return the size of the target"""
        return self.train.action_space

    def _make_env(self, start=None, size=None):
        return ParallelEnvironment(
            self.parallel_env,
            self.transforms,
            gym.make,
            self.env_name,
        )

    @property
    def train(self):
        if self._train is None:
            self._train = self._make_env(self._start_train(), self.train_size)
        return self._train

    @property
    def valid(self):
        if self._valid is None:
            self._valid = self._make_env(self._start_valid(), self.valid_size)
        return self._valid

    @property
    def test(self):
        if self._test is None:
            self._test = self._make_env(self._start_test(), self.test_size)
        return self._test

    @staticmethod
    def categories():
        """Dataset tags so we can filter what we want depending on the task"""
        return set(['RL'])

    def state_dict(self):
        return {}

    def load_state_dict(self, data):
        pass

    def sample_action(self):
        return np.ones(self.parallel_env) * self.train.action_space.sample()

    def max(self):
        return self.train


builders = {
    env: functools.partial(GymEnvironment, env_name=env) for env in env_registry.env_specs.keys()
}
builders['gym'] = GymEnvironment
