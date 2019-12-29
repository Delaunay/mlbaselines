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


     >>> * <------------------- steps --------------------------------->
     >>> ^ [states 0] [states 1] [states 2] [states 3]
     >>> | [states 0] [states 1] [states 2]
     >>> | [states 0] [states 1] [states 2] [states 3]
     >>> v [states 0] [states 1] [states 2] [states 3] [states 4]
     >>> * <------------------- steps --------------------------------->
     >>>     Batch 0    Batch 1    Batch 2    Batch 3    Batch 4

    Notes
    -----
    Steps:
        Number of Simulation Steps

    Simulation:
        Number of parallel simulation

    Examples
    --------
    The output below shows the size of each fields with ``num_steps=32``
    ``num_simulation=4`` and with a state size of ``3, 210, 160`` (images of the simulation)

    >>> replay.describe()
    >>> rewards      : torch.Size([32, 4])
    >>> states       : torch.Size([32, 4, 3, 210, 160])
    >>> next_states  : torch.Size([32, 4, 3, 210, 160])
    >>> critic_values: torch.Size([32, 4])
    >>> actions      : torch.Size([32, 4])
    >>> log_probs    : torch.Size([32, 4])
    >>> mask         : torch.Size([32, 4])
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

    def to_dict(self):
        return {
            'state': self.states(),
            'new_state': self.next_states(),
            'action': self.actions(),
            'log_prob': self.log_probs(),
            'entropy': self.entropies(),
            'critic': self.critic_values(),
            'reward': self.rewards(),
            'mask': self.masks(),
        }

    def describe(self):
        print('rewards      :', self.rewards().shape)
        print('states       :', self.states().shape)
        print('next_states  :', self.next_states().shape)
        print('critic_values:', self.critic_values().shape)
        print('actions      :', self.actions().shape)
        print('log_probs    :', self.log_probs().shape)
        print('mask         :', self.masks().shape)
        print('entropy      :', self.entropies().shape)

    def append(self, transition: Transition):
        self.state_size = transition.state.shape[1]
        self.simulation_batch = transition.state.shape[0]
        self.transitions.append(transition)

    def _accumulate_transitions(self, by):
        return [getattr(t, by) for t in self.transitions]

    def actions(self):
        """
        Returns
        -------
        A tensor of the action that was taken (Steps, Sim, 1)
        """
        return torch.stack(self._accumulate_transitions('action'))

    def rewards(self):
        r = torch.stack(self._accumulate_transitions('reward'))
        self.grad_batch = r.shape[0]
        return r

    def log_probs(self):
        return torch.stack(self._accumulate_transitions('log_prob'))

    def entropies(self):
        return torch.stack(self._accumulate_transitions('entropy'))

    def critic_values(self):
        return torch.stack(self._accumulate_transitions('critic'))

    def masks(self):
        return torch.stack(self._accumulate_transitions('mask'))

    def states(self):
        """
        Returns
        -------
        A tensor of the simulation states (Steps, Sim, State size...)
        """
        return torch.stack(self._accumulate_transitions('state'))

    def next_states(self):
        """
        Returns
        -------
        A tensor of the simulation states (Steps, Sim, State size...)
        """
        return torch.stack(self._accumulate_transitions('next_state'))
