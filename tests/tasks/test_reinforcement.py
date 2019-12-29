from functools import reduce
import time
import gym

import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision.transforms as transforms
import torch.nn.functional as F

from olympus.reinforcement.gymenv import RLDataloader
from olympus.tasks.rl import ReinforcementLearningA2C, ActorCritic


# Return the best prob action to take
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

    def forward(self, x):
        x = x.view(-1, self.state_space)
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


# return a State value
class CriticRAM(nn.Module):
    """Critic that takes in input the RAM/vector as the state of the Game"""
    def __init__(self, state_shape):
        super(CriticRAM, self).__init__()
        state_space = 1
        for dim in state_shape:
            state_space *= dim

        self.state_space = state_space
        self.affine1 = nn.Linear(state_space, 128)
        self.affine2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(-1, self.state_space)
        x = F.relu(self.affine1(x))
        return self.affine2(x)


class ConvNet(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=32):
        super(ConvNet, self).__init__()
        self.action_space = action_space

        # build network & check size are consistent
        with torch.no_grad():
            # batch size = 1
            tracer = torch.randn((1,) + state_space)

            self.conv = nn.Sequential(
                nn.Conv2d(3, 20, 5, 1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(20, 50, 5, 1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )

            tracer = self.conv(tracer)
            self.conv_output_size = reduce(
                lambda x, y: x * y, tracer.shape[1:], 1)

            tracer = tracer.view(-1, self.conv_output_size)

            self.output = nn.Sequential(
                nn.Linear(self.conv_output_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_space),
                nn.LogSoftmax(dim=1)
            )
            _ = self.output(tracer)

    def forward(self, x):
        x = self.conv(x)
        return self.output(x.view(-1, self.conv_output_size))


ActorConv = ConvNet
CriticConv = ConvNet

def to_nchw(states):
    return states.permute(0, 3, 1, 2)


loader = RLDataloader(
     8,  # Number of parallel simulations
    200,  # Max number of steps in a simulation
    # transform state
    transforms.Lambda(to_nchw),
    gym.make,
    'SpaceInvaders-v0'  # Simulation Name Vision
    # 'Enduro-ram-v0'   # Simulation Name RAM
)

print('loader.state_vector_shape: ', loader.state_vector_shape)
print('loader.action_vector_size: ', loader.action_vector_size)


actor = ActorConv(
    loader.state_vector_shape,
    loader.action_vector_size
)

critic = CriticConv(
    loader.state_vector_shape,
    1
)

actor_critic = ActorCritic(
    actor,
    critic
)

task = ReinforcementLearningA2C(
    actor_critic=actor_critic,
    optimizer=Adam(actor_critic.parameters(), lr=1e-2),
    gamma=0.99,
    num_steps=4
)

task.device = torch.device('cpu')

# smooth_loss = ExponentialSmoothing(0.90)
epoch = 10
for i in range(0, epoch):
    frame_count = task.frame_count
    s = time.time()
    loss = task.fit(i, loader.iterator(), None)
    elapsed = time.time() - s
    frame_count = task.frame_count - frame_count

    fps = frame_count / elapsed
    print(f'Epoch [{i:4d}/{epoch:4d}] loss={loss:8.4f}  frames={frame_count:5d} fps={fps:8.4f} ' 
          f'batch={task.batch_size}')

report = loader.env.report()
import json
print(json.dumps(report, indent=2))

