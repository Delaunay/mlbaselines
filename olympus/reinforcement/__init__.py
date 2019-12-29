from olympus.utils import warning
from olympus.utils.factory import fetch_factories

registered_environment = fetch_factories('olympus.reinforcement', __file__)


def known_environments(*category_filters, include_unknown=False):
    if not category_filters:
        return registered_environment.keys()

    matching = []
    for filter in category_filters:
        for name, factory in registered_environment.items():

            if hasattr(factory, 'categories'):
                if filter in factory.categories():
                    matching.append(name)

            # we don't know if it matches because it does not have the categories method
            elif include_unknown:
                matching.append(name)

    return matching


def register_environment(name, factory, override=False):
    global registered_environment

    if name in registered_environment:
        warning(f'{name} was already registered, use override=True to ignore')

        if not override:
            return

    registered_environment[name] = factory


class RegisteredEnvironmentNotFound(Exception):
    pass


class Environment:
    def __init__(self, env_name, transforms=None, rand_seed=None, train_size=1024, valid_size=128, test_size=128, parallel_env=8,
                 num_thread=4,
                 distribution_mode='easy'):

        env_ctor = registered_environment.get(env_name)

        if env_ctor is None:
            raise RegisteredEnvironmentNotFound(env_name)

        self.env = env_ctor(
            transforms=transforms,
            rand_seed=rand_seed,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            parallel_env=parallel_env,
            num_thread=num_thread,
            distribution_mode=distribution_mode)

    def close(self):
        return self.env.close()

    @property
    def state_space(self):
        return self.input_size

    @property
    def action_space(self):
        return self.target_size

    @property
    def input_size(self):
        """Return the size of the samples"""
        return self.env.input_size

    @property
    def target_size(self):
        """Return the size of the target"""
        return self.env.action_space

    @property
    def train(self):
        return self.env.train

    @property
    def valid(self):
        return self.env.valid

    @property
    def test(self):
        return self.env.test

    def categories(self):
        """Dataset tags so we can filter what we want depending on the task"""
        return self.env.categories

    def state_dict(self):
        return self.env.state_dict()

    def load_state_dict(self, data):
        return self.env.load_state_dict(data)

    def sample_action(self):
        return self.env.sample_action()

    def max(self):
        return self.env.train
