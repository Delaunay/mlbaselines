from collections import namedtuple
import torch
from typing import List

Transition = namedtuple('Transition', [
    'state',        # Game State
    'action',       # Action performed by the network
    'reward',       # Reward received for the action
    'log_prob',     # Log_prob of the action
    'entropy',      # Action entropy
    'critic',       # Critic Value of the action
    'mask',         # Mask is not done ?
    'next_state'    # New Game State resulting from the Action
])


class ReplayVector:
    """Holds all the state transition of the simulation for training purposes

    Attributes
    ----------
    transitions:
        List of all the stored transitions

    state_size:
        Size of the simulation state

    simulation_batch:
        Number of different simulation state in one Transition Struct

    grad_batch:
        Total number of states in this object `grad_batch = simulation_batch * len(transitions)`
    """

    __slots__ = (
        'transitions', 'state_size', 'simulation_batch', 'grad_batch'
    )

    def __init__(self):
        self.state_size = None
        self.simulation_batch = None
        self.grad_batch = None
        self.transitions: List[Transition] = []

    def __len__(self):
        return len(self.transitions)

    def __bool__(self):
        return len(self.transitions) > 0

    def __iter__(self):
        return iter(self.transitions)

    def __reversed__(self):
        return reversed(self.transitions)

    def append(self, transition: Transition):
        self.state_size = transition.state.shape[1]
        self.simulation_batch = transition.state.shape[0]
        self.transitions.append(transition)

    def _accumulate_transitions(self, by):
        return [getattr(t, by) for t in self.transitions]

    def actions(self):
        return torch.cat(self._accumulate_transitions('action')).view(-1, 1)

    def rewards(self):
        r = torch.cat(self._accumulate_transitions('reward')).view(-1, 1)
        self.grad_batch = r.shape[0]
        return r

    def log_probs(self):
        return torch.cat(self._accumulate_transitions('log_prob')).view(-1, 1)

    def entropies(self):
        return torch.cat(self._accumulate_transitions('entropy')).view(-1, 1)

    def critic_values(self):
        return torch.cat(self._accumulate_transitions('critic')).view(-1, 1)

    def masks(self):
        return torch.cat(self._accumulate_transitions('mask')).view(-1, 1)

    def states(self):
        return torch.cat(self._accumulate_transitions('state'))

    def next_states(self):
        return torch.cat(self._accumulate_transitions('next_state'))
