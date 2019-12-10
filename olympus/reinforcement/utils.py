from collections import namedtuple

from torch.distributions import Categorical, Normal
from torch.nn import Module


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


# https://sergioskar.github.io/Actor_critics/
# Value Based: They try to find or approximate the optimal value function, which is a mapping between an action and a value. The higher the value, the better the action. The most famous algorithm is Q learning and all its enhancements like Deep Q Networks, Double Dueling Q Networks, etc
# Policy-Based: Policy-Based algorithms like Policy Gradients and REINFORCE try to find the optimal policy directly without the Q -value as a middleman.


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

    def critic(self, states):
        return self.forward(states)

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
