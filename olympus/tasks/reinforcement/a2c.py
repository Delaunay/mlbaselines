from typing import Callable

import numpy as np
import torch
from torch.distributions import Categorical, Distribution
from torch.nn import Module
from torch.optim import Optimizer

from olympus.tasks.task import Task
from olympus.utils import select, drop_empty_key
from olympus.reinforcement.utils import AbstractActorCritic
from olympus.resuming import state_dict, load_state_dict, BadResumeGuard
from olympus.observers import ProgressView, Speed, ElapsedRealTime, CheckPointer
from olympus.metrics.named import NamedMetric


class A2C(Task):
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
    def __init__(self, model: AbstractActorCritic, dataloader, optimizer, lr_scheduler, device, criterion=None,
                 storage=None, logger=None):
        super(A2C, self).__init__(device=device)

        if criterion is None:
            criterion = lambda x: x.sum()

        self.actor_critic = model
        self.lr_scheduler = lr_scheduler
        self.optimizer: Optimizer = optimizer
        self.criterion: Module = criterion
        self.gamma: float = 0.99
        self.eps = np.finfo(np.float32).eps.item()
        self.action_sampler: Callable[[], Distribution] = Categorical
        self.tensor_shape = None
        self.frame_count: int = 0
        self.dataloader = dataloader
        self.storage = storage
        self._first_epoch = 0
        self.current_epoch = 0

        self.metrics.append(NamedMetric(name='loss'))
        self.metrics.append(ElapsedRealTime())
        self.metrics.append(Speed())
        self.metrics.append(ProgressView(speed_observer=self.metrics.get('Speed')))

        if storage:
            self.metrics.append(CheckPointer(storage=storage))

        self.hyper_parameters = {}
        self.batch_size = None

    # Hyper Parameter Settings
    # ---------------------------------------------------------------------
    def get_space(self, **fidelities):
        """Return hyper parameter space"""
        return drop_empty_key({
            'optimizer': self.optimizer.get_space(),
            'lr_schedule': self.lr_scheduler.get_space(),
            'model': self.actor_critic.get_space(),
            'gamma': 'loguniform(0.99, 1)'
        })

    def init(self, gamma=0.99, optimizer=None, lr_schedule=None, model=None, uid=None):
        """
        Parameters
        ----------
        optimizer: Dict
            Optimizer hyper parameters

        lr_schedule: Dict
            lr schedule hyper parameters

        model: Dict
            model hyper parameters

        gamma: float
            reward discount factor

        trial: Optional[str]
            trial id to use for logging.
            When using orion usually it already created a trial for us we just need to append to it
        """

        optimizer = select(optimizer, {})
        lr_schedule = select(lr_schedule, {})
        model = select(model, {})
        self.gamma = gamma

        self.actor_critic.init(
            **model
        )

        # We need to set the device now so optimizer receive cuda tensors
        self.set_device(self.device)
        self.optimizer.init(
            self.actor_critic.parameters(),
            override=True, **optimizer
        )
        self.lr_scheduler.init(
            self.optimizer,
            override=True, **lr_schedule
        )

        self.hyper_parameters = {
            'optimizer': optimizer,
            'lr_schedule': lr_schedule,
            'model': model
        }

        parameters = {}
        parameters.update(optimizer)
        parameters.update(lr_schedule)
        parameters.update(model)

        self.metrics.new_trial(parameters, uid)
        self.set_device(self.device)

    # Training
    # --------------------------------------------------------------------
    @property
    def _epoch(self):
        return int(self.dataloader.completed_simulations)

    def fit(self, epochs, context=None):
        self._fix()

        with BadResumeGuard(self):
            self._start(epochs)

            prev = self._epoch
            step = 0

            for _, vector in enumerate(self.dataloader):
                last_state = self.dataloader.state
                self.metrics.new_batch(step, vector)

                results = self.advantage_actor_critic(last_state, vector)
                self.metrics.end_batch(step, vector, results)

                step += 1

                # Called every time a simulation gets completed
                if self._epoch > prev:
                    self.metrics.new_epoch(self._epoch, None)
                    self.lr_scheduler.epoch(self._epoch)

                    prev = self._epoch
                    step = 0

                # Epochs is the number of completed simulations
                if self.dataloader.completed_simulations + 1 >= epochs:
                    break

            self.report(pprint=True, print_fun=print)

        self.metrics.end_train()
        # Free all the file descriptor opened by the Gym envs
        self.dataloader.close()

    def compute_returns(self, value, states):
        reward = value
        returns = []

        for state in reversed(states):
            reward = state.reward + self.gamma * reward * state.mask
            returns.insert(0, reward)

        return returns

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
        returns = torch.stack(returns)

        log_probs = replay_vector.log_probs()
        values = replay_vector.critic_values()
        advantage = returns - values

        actor_loss = -(log_probs * advantage).mean()
        critic_loss = advantage.pow(2).mean()

        self.tensor_shape = tuple([int(i) for i in advantage.shape])
        self.batch_size = int(self.tensor_shape[0])
        self.frame_count += self.batch_size

        entropy = 0
        for action in replay_vector:
            entropy += action.entropy

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        results = {
            'loss': loss.item(),
            'actor': actor_loss.item(),
            'critic': critic_loss.item()
        }
        return results

    # CheckPointing
    # ---------------------------------------------------------------------
    def load_state_dict(self, state, strict=True):
        load_state_dict(self, state, strict, force_default=True)
        self._first_epoch = state['epoch']
        self.current_epoch = state['epoch']

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = state_dict(self, destination, prefix, keep_vars, force_default=True)
        state['epoch'] = self.current_epoch
        return state

    @property
    def model(self):
        return self.actor_critic

    @model.setter
    def model(self, model):
        self.actor_critic = model

    def parameters(self):
        return self.actor_critic.parameters()
