import pytest

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torchvision.transforms as transforms

import gym

from olympus.reinforcement.dataloader import RLDataloader


def to_nchw(states):
    return states.permute(0, 3, 1, 2)


environments = [
    'SpaceInvaders-v0',
    'Enduro-ram-v0'
]


workers = [1] # , 2


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
        x = x.view(-1, self.state_space)
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def act(self, x):
        return torch.argmax(self.forward(x))


@pytest.mark.parametrize('worker', workers)
@pytest.mark.parametrize('env_name', environments)
def test_parallel_environment(env_name, worker):
    loader = RLDataloader(
        worker,         # Number of parallel simulations
        200,            # Max number of steps in a simulation
        # transform state
        transforms.Lambda(to_nchw),
        gym.make,
        env_name
    )

    iter = loader.iterator()
    actor = ActorRAM(loader.state_vector_shape, loader.action_vector_size)

    state = iter.next(action=None)

    for i in range(0, 10):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)

        print(state)
        action = actor.act(state.to(dtype=torch.float32))
        print(action)
        state, reward, done, _ = iter.next(action.detach())

