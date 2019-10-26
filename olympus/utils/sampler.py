import torch
from torch.random import fork_rng
from torch.utils.data.sampler import Sampler


class ResumableSampler(Sampler):
    """Cannot restart mid epoch,
    this does not work because get_rng_state does not return the correct after sampling state"""

    def __init__(self, sampler: Sampler, seed: int = None, state=None):
        super(ResumableSampler, self).__init__(data_source=None)
        self.sampler = sampler
        self.seed = seed

        if state is not None:
            self.rng_state = state

        elif seed is not None:
            with fork_rng():
                torch.manual_seed(seed)
                self.rng_state = torch.get_rng_state()

        else:
            self.rng_state = torch.get_rng_state()

    def __iter__(self):
        with fork_rng() as rng:
            torch.set_rng_state(self.rng_state)
            iterator = iter(self.sampler)
            self.rng_state = torch.get_rng_state()
            print(self.rng_state, rng)
            return iterator

    def __len__(self):
        return len(self.sampler)

    def load_state_dict(self, state):
        rng_state = state.get('rng_state')
        self.rng_state = rng_state

    def state_dict(self):
        return {
            'rng_state': self.rng_state
        }
