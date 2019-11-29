"""
https://sergioskar.github.io/Actor_critics/

    Value Based: They try to find or approximate the optimal value function, which is a mapping between an action and a value. The higher the value, the better the action. The most famous algorithm is Q learning and all its enhancements like Deep Q Networks, Double Dueling Q Networks, etc

    Policy-Based: Policy-Based algorithms like Policy Gradients and REINFORCE try to find the optimal policy directly without the Q -value as a middleman.

"""
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.distributions import Categorical, Distribution, Normal
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


from olympus.tasks.task import Task
from olympus.metrics import MetricList
from olympus.utils import get_value
from olympus.reinforcement.replay import ReplayVector


# https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
# Actor Critic
# Policy Gradient
# A3C   : Asynchronous Advantage Actor-Critic
# A2C   :  Synchronous Advantage Actor-Critic <- Better GPU utilisation
# DPG   : Deterministic policy gradient
# DDPG  : Deep Deterministic Policy Gradient,
# DQN   : Deep Q-Network
# D4PG  : Distributed Distributional DDPG
# MADDPG: Multi-agent DDPG
# TRPO  : Trust region policy optimization
# PPO   : proximal policy optimization
# ACER  : actor-critic with experience replay
# ACTKR : actor-critic using Kronecker-factored trust region
# SAC   : Soft Actor-Critic
# TD3   : Twin Delayed Deep Deterministic
#


class AbstractActorCritic(Module):
    def act(self, state):
        raise NotImplementedError()

    def critic(self, state):
        raise NotImplementedError()

    def forward(self, *input):
        assert False, 'Forward should not be called'

    def actor_parameters(self):
        raise NotImplementedError()

    def critic_parameters(self):
        raise NotImplementedError()


class ActorCriticDistributional(AbstractActorCritic):
    """The actor returns the parameter of a distribution

    References
    ----------
    .. [1] https://arxiv.org/abs/1707.06887
    """

    def __init__(self, actor, critic, distribution=Normal):
        super(ActorCriticDistributional, self).__init__()

        self._critic = critic
        self._actor = actor
        self.distribution = distribution

    def act(self, state):
        distribution_parameters = self._actor(state)
        dist = self.distribution(*distribution_parameters)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(1)
        return action, log_prob, dist.entropy().mean()

    def critic(self, state):
        return self._critic(state)

    def actor_parameters(self):
        return self._actor.parameters()

    def critic_parameters(self):
        return self._critic.parameters()


class ActorCriticCategorical(AbstractActorCritic):
    """The actor returns the probabilities of each actions"""

    def __init__(self, actor, critic):
        super(ActorCriticCategorical, self).__init__()

        self._critic: Module = critic
        self._actor: Module = actor
        self.action_sampler = Categorical

    def act(self, state):
        probabilities = self._actor(state)
        dist = self.action_sampler(probabilities)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(1)
        return action.clamp(-2, 2), log_prob, dist.entropy().mean()

    def critic(self, state):
        return self._critic(state)

    def actor_parameters(self):
        return self._actor.parameters()

    def critic_parameters(self):
        return self._critic.parameters()


ActorCritic = ActorCriticCategorical

# Replay Vector
SavedAction = namedtuple('SavedAction', [
    'state',        # Game State
    'action',       # Action performed by the network
    'reward',       # Reward received for the action
    'log_prob',     # Log_prob of the action
    'entropy',      # Action entropy
    'critic',       # Critic Value of the action
    'mask',         # Mask is not done ?
    'next_state'    # New Game State resulting from the Action
])


@dataclass
class ReinforcementLearningA2C(Task):
    """

    Parameters
    ----------
    actor_critic: Module
        Torch Module that takes a state and return an action and a value

    env: Env
        Gym like environment

    num_steps: int
        number of simulation/environment steps to accumulate before doing a gradient step

    Notes
    -----

    RL has two batch size, the data loader batch size (lbs) which is equivalent to the number
    of simulation done in parallel and the gradient batch size.

    num_steps of simulations are accumulated together to perform one gradient update

    """
    actor_critic: AbstractActorCritic
    optimizer: Optimizer
    criterion: Module = lambda x: x.sum()
    metrics = MetricList()
    num_steps: int = 5
    gamma: float = 0.99
    eps = np.finfo(np.float32).eps.item()
    action_sampler: Callable[[], Distribution] = Categorical
    batch_size: int = None
    tensor_shape = None
    frame_count: int = 0

    def compute_returns(self, value, actions):
        reward = value
        returns = []

        for action in reversed(actions):
            reward = action.reward + self.gamma * reward * action.mask
            returns.insert(0, reward)

        return returns

    def ppo(self, current_state, replay_vector, ppo_epoch=5, ppo_batch_size=32, ppo_clip_param=10, ppo_max_grad_norm=1000):
        """New policy gradient methods for reinforcement learning, which alternate  between  sampling  data
        through  interaction  with  the  environment,  and  optimizing  a“surrogate” objective function
        using stochastic gradient ascent.
        Whereas standard policy gradient  methods  perform  one  gradient  update  per  data  sample,
        we  propose  a  novel  objective function that enables multiple epochs of mini-batch updates.

        References
        ----------
        Original Paper https://arxiv.org/pdf/1707.06347.pdf

        """

        # Accumulate all the observation into tensors we can sample
        # state.shape: (# parallel env * self.num_steps) x (state vector size)
        state       = replay_vector.states().detach()
        next_state  = replay_vector.next_states().detach()

        # action.shape: (# parallel env * self.num_steps) x 1
        # action      = torch.cat([t.action for t in replay_vector]).view(-1, 1)
        reward      = replay_vector.rewards().detach()
        log_prob    = replay_vector.log_probs().detach()

        # Normalize rewards
        reward = (reward - reward.mean()) / (reward.std() + 1e-10)

        with torch.no_grad():
            target_v = reward + self.gamma * self.actor_critic.critic(next_state)

        advantage = (target_v - self.actor_critic.critic(state)).detach()
        all_loss = 0

        self.batch_size = ppo_batch_size
        for p in range(ppo_epoch):
            sampler = BatchSampler(
                sampler=SubsetRandomSampler(range(len(state))),
                batch_size=ppo_batch_size,
                drop_last=True
            )

            epoch_loss = 0
            count = 0

            for count, indices in enumerate(sampler):
                state_batch = state[indices]
                action, new_log_prob, entropy = self.actor_critic.act(state_batch)

                ratio = torch.exp(new_log_prob - log_prob[indices])

                advantage_batch = advantage[indices]
                L1 = ratio * advantage_batch
                L2 = torch.clamp(ratio, 1 - ppo_clip_param, 1 + ppo_clip_param) * advantage_batch

                action_loss = -torch.min(L1, L2).mean()

                value_loss = F.smooth_l1_loss(self.actor_critic.critic(state_batch), target_v[indices])

                # Harmonize cost function with A2C implementation
                # self.actor_optimizer.zero_grad()
                # action_loss.backward()
                # nn.utils.clip_grad_norm_(self.actor_net.parameters(), ppo_max_grad_norm)
                # self.actor_optimizer.step()
                #
                # self.critic_net_optimizer.zero_grad()
                # value_loss.backward()
                # nn.utils.clip_grad_norm_(self.critic_net.parameters(), ppo_max_grad_norm)
                # self.critic_net_optimizer.step()

                self.optimizer.zero_grad()
                loss = action_loss + 0.5 * value_loss - 0.001 * entropy
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), ppo_max_grad_norm)
                self.optimizer.step()

                epoch_loss += loss
                self.frame_count += self.batch_size

            epoch_loss /= count + 1
            all_loss += epoch_loss

        return all_loss / ppo_epoch

    def advantage_actor_critic(self, current_state, replay_vector):
        """A2C Synchronous actor Critic

        Parameters
        ----------
        current_state:
            current state the game was left in

        replay_vector:
            list of action that was performed by the model to reach current state
        """
        value = self.actor_critic.critic(current_state)
        returns = self.compute_returns(value, replay_vector)

        log_probs = replay_vector.log_probs()
        returns = torch.cat(returns).detach()
        values = replay_vector.critic_values()

        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        self.tensor_shape = advantage.shape
        self.batch_size = self.tensor_shape[0]

        entropy = 0
        for action in replay_vector:
            entropy += action.entropy

        self.frame_count += self.batch_size
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach()

    def fit(self, step, dataloader, context):
        # starting epoch
        state = dataloader.next(action=None)
        state = state.to(self.device)

        is_done = False
        all_loss = []

        while not is_done:
            replay_vector = ReplayVector()

            # Accumulates simulation steps in batches
            for _ in range(self.num_steps):
                action, log_prob, entropy = self.actor_critic.act(state)
                critic = self.actor_critic.critic(state)

                new_state = dataloader.next(action)
                if new_state is not None:
                    old_state = state
                    state, reward, done, _ = new_state
                    state = state.to(self.device)

                    # Make sure all Tensor follows the size below
                    #   (WorkerSize x size) using unsqueeze
                    replay_vector.append(SavedAction(
                        action      = action,
                        reward      = reward.unsqueeze(1).to(device=self.device),
                        log_prob    = log_prob.unsqueeze(1),
                        entropy     = entropy,
                        critic      = critic,
                        mask        = (1 - done).unsqueeze(1).to(device=self.device),
                        state       = old_state,
                        next_state  = state
                    ))
                else:
                    is_done = True
                    break

            # optimize for current batch
            if replay_vector:
                # loss = self.advantage_actor_critic(state, replay_vector)
                loss = self.ppo(current_state=state, replay_vector=replay_vector)
                all_loss.append(get_value(loss))

        # epoch is done compute the epoch loss
        sum_loss = 0
        for loss in all_loss:
            sum_loss += get_value(loss)

        return sum_loss / len(all_loss)
