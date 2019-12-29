import functools

import numpy as np
from procgen import ProcgenEnv
from procgen.env import ENV_NAMES

import torch


class ProcEnvAdapter:
    def __init__(self, proc_env, transforms=None):
        self.env = proc_env
        self.transforms = transforms
        if self.transforms is None:
            self.transforms = lambda x: x

    def __getattr__(self, item):
        if hasattr(self.env, item):
            return getattr(self.env, item)
        raise AttributeError(f'{item} is not an attribute of ProcEnvAdapter')

    def step(self, *args, **kwargs):
        obs, rew, done, info = self.env.step(*args, **kwargs)
        return self.transforms(self._torch(obs.get('rgb'))), rew, done, info

    def reset(self):
        obs = self.env.reset()
        return self.transforms(self._torch(obs.get('rgb')))

    def _torch(self, val):
        return torch.from_numpy(val)


class ProcgenEnvironment:
    def __init__(self, env_name, transforms=None, rand_seed=None, train_size=1024, valid_size=128, test_size=128, parallel_env=8, num_thread=4,
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

        shape = 64, 64, 3
        input = torch.randn((1,) + shape)
        self.input_shape = self.transforms(input).shape[1:]

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
        return self.input_shape

    @property
    def target_size(self):
        """Return the size of the target"""
        return 15,

    def _make_env(self, start, size):
        return ProcEnvAdapter(ProcgenEnv(
            num_envs=self.parallel_env,
            start_level=start,
            num_levels=size,
            env_name=self.env_name,
            num_threads=self.num_thread,
            rand_seed=self.seed
        ), self.transforms)

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
    env: functools.partial(ProcgenEnvironment, env_name=env) for env in ENV_NAMES
}
builders['procgen'] = ProcgenEnvironment
