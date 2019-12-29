import pytest

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import gym


import sys
sys.stderr = sys.stdout


def to_nchw(states):
    if isinstance(states, np.ndarray):
        states = torch.from_numpy(states)

    # Make sure batch_size of 1 is represented in the shape
    if len(states.shape) == 3:
        states = states.unsqueeze(0)

    # if it is not an image it has less than 3 channels
    elif len(states.shape) != 4:
        return states

    states = states.permute(0, 3, 1, 2)
    return states


environments = [
    'SpaceInvaders-v0',
    'Enduro-ram-v0'
]


workers = [1, 2]


class ActorRAM(nn.Module):
    """Actor that takes in input the RAM/vector as the state of the Game"""
    def __init__(self, state_shape, action_space):
        super(ActorRAM, self).__init__()
        state_space = 1
        for dim in state_shape:
            state_space *= dim

        self.state_space = state_space
        self.affine1 = nn.Linear(state_space, 128)
        self.affine2 = nn.Linear(128, action_space)

        self.dist = Normal

    def forward(self, x):
        x = x.reshape(-1, self.state_space)
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def act(self, x):
        return torch.argmax(self.forward(x), dim=1)


# @pytest.mark.parametrize('worker', workers)
# @pytest.mark.parametrize('env_name', environments)
# def test_environment(env_name, worker):
#     loader = RLDataloader(
#         worker,         # Number of parallel simulations
#         200,            # Max number of steps in a simulation
#         # transform state
#         to_nchw,
#         gym.make,
#         env_name
#     )
#
#     iter = loader.iterator()
#     actor = ActorRAM(loader.state_vector_shape, loader.action_vector_size)
#     state = iter.next(action=None)
#
#     for i in range(0, 1):
#         action = actor.act(state.to(dtype=torch.float32))
#         state, reward, done, _ = iter.next(action.detach())
#
