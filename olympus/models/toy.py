from functools import reduce

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ConvNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(ConvNet, self).__init__()
        self.action_space = output_size
        self.state_space = input_size
        self.distribution = Categorical

        # build network & check size are consistent
        with torch.no_grad():
            # batch size = 1
            tracer = torch.randn((1,) + self.state_space)

            self.conv = nn.Sequential(
                nn.Conv2d(3, 20, 5, 1),
                nn.BatchNorm2d(20),
                nn.ReLU()
            )

            tracer = self.conv(tracer)

            self.conv_output_size = reduce(
                lambda x, y: x * y, tracer.shape[1:], 1)
            tracer = tracer.view(-1, self.conv_output_size)

            self.output = nn.Sequential(
                nn.Linear(self.conv_output_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.action_space),
                nn.Softmax(dim=1),
            )

            _ = self.output(tracer)

    def forward(self, x):
        x = x / 255
        x = self.conv(x)
        return self.output(x.view(-1, self.conv_output_size))

    def act(self, states):
        distribution_parameters = None
        try:
            distribution_parameters = self.forward(states)
            dist = self.distribution(distribution_parameters)
            action = dist.sample()
            log_prob = dist.log_prob(action).unsqueeze(1)

            return action, log_prob, dist.entropy()
        except:
            print(distribution_parameters)
            raise

    def critic(self, states):
        return self.forward(states).sum(dim=1)


def convnet(input_size, output_size):
    return ConvNet(input_size, output_size)


builders = {
    'toy_rl_convnet': convnet
}

