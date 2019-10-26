import random
import time
from torch.utils.data.sampler import Sampler


class _BaseSampler(Sampler):
    """Standalone Sampler that does not use a global PRNG.
    This makes the results more reproducible as no outside call can modify the state of this sampler.
    """
    def __init__(self, seed):
        if seed is None:
            seed = time.time()
        self.rand_engine = random.Random(seed)

    def get_state(self):
        return self.rand_engine.getstate()

    def set_state(self, state):
        return self.rand_engine.setstate(state)


class RandomSamplerWithoutReplacement(_BaseSampler):
    """Permute the list of items"""
    def __init__(self, data_source, seed):
        super(RandomSamplerWithoutReplacement, self).__init__(seed)
        self.data_source = data_source
        self.draws = list(range(len(self.data_source)))

    def _generate_draws(self):
        self.rand_engine.shuffle(self.draws)
        return self.draws

    def __iter__(self):
        return iter(self._generate_draws())

    def __len__(self):
        return len(self.data_source)


class RandomSamplerWithReplacement(_BaseSampler):
    def __init__(self, data_source, num_samples, seed):
        super(RandomSamplerWithReplacement, self).__init__(seed)

        self.data_source = data_source
        self.num_samples = num_samples

    def __iter__(self):
        draws = [self.rand_engine.randrange(start=0, stop=len(self)) for _ in self.num_samples]
        return iter(draws)

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    """Pytorch like Sampler"""
    def __init__(self, data_source, seed=None, replacement=False, num_samples=None):
        self.sampler = None

        if num_samples is None:
            num_samples = len(data_source)

        if replacement:
            self.sampler = RandomSamplerWithReplacement(data_source, num_samples, seed)
        else:
            self.sampler = RandomSamplerWithoutReplacement(data_source, seed)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return len(self.sampler)

