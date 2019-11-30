from olympus.metrics.metric import Metric
from typing import List

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from gym import Env

# check this out
# https://arxiv.org/abs/1709.06560


class Validation(Metric):
    """Generates random trajectories to validate our model on"""

    def __init__(self, env, trajectory_length, seeds=None, batch_size=128, throws=1000):
        # Generate Validation Trajectories
        action_space = env.action_space()
        self.seeds = seeds

        env.seed(seeds)
        state = env.reset()
        states = [state]
        dones = [None]
        actions = []

        for _ in range(trajectory_length):
            action = np.random.randint(0, action_space)
            state, reward, done = env.step(action)

            actions.append(action)
            states.append(state)
            dones.append(done)

        self.actions = torch.cat(actions)
        self.states = torch.cat(states)
        self.masks = 1 - torch.cat(dones)
        self.env = env
        self.sampler = BatchSampler(
            sampler=SequentialSampler(range(len(self.states))),
            batch_size=batch_size,
            drop_last=True
        )

    def compute_rewards(self, task):
        self.env.seed(self.seeds)
        state = self.env.reset()

        for batch in self.sampler:
            states = self.states[batch]
            masks = self.masks[batch]
            action = self.actions[batch]

            action, new_log_prob, entropy = task.actor_critic.act(states)
            values = task.actor_critic.critic(states)

            new_states, reward, done = env.step(action)




class ReinforcementTest(Metric):
    """Compute average and sd reward of a given model, env over a given number of plays"""

    def __init__(self, env, model, vis=False, epsilon=0.01):
        self.env: Env = env
        self.vis: bool = vis
        self.device = torch.device('cpu')
        self.model = model
        self.rewards: List = []
        self.sd: List = []
        self.sample_count = 10
        self.epsilon = epsilon
        self.action_space = self.env.action_space()

    def on_new_epoch(self, step, task, input, context):
        return self.compute_rewards(task)

    def compute_rewards(self, task):
        rewards = [self.compute_reward(task) for _ in range(self.sample_count)]
        self.rewards.append(np.mean(rewards))
        self.sd.append(np.std(rewards))

    def epsilon_act(self, task, state):
        """Do a random action from time to time to shake things up"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_space)

        return task.actor_critic.act(state)

    def compute_reward(self, task):
        done = False
        total_reward = 0

        task.actor_critic.eval()
        with torch.no_grad():
            state = self.env.reset()

            if self.vis:
                self.env.render()

            while not done:
                state = (torch.FloatTensor(state)
                         .unsqueeze(0)
                         .to(self.device))

                action, _, _ = self.epsilon_act(task, state)

                next_state, reward, done, _ = self.env.step(action)
                state = next_state

                if self.vis:
                    self.env.render()

                total_reward += reward

        task.actor_critic.train()
        return total_reward

    def finish(self, task):
        self.compute_rewards(task)

    def value(self):
        return {
            'test_mean_reward': self.rewards[-1],
            'test_mean_sd': self.sd[-1],
            'test_sample_count': self.sample_count
        }

    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.rewards, 'b-')
        plt.title('{}. reward: {}'.format(len(self.rewards), self.rewards[-1]))
        plt.pause(0.0001)


