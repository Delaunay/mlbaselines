from typing import Callable

import numpy as np
import torch
from torch.distributions import Categorical, Distribution
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import BatchSampler, SubsetRandomSampler

from olympus.tasks.task import Task
from olympus.utils import select
from olympus.reinforcement.utils import AbstractActorCritic
from olympus.resuming import state_dict, load_state_dict, BadResumeGuard
from olympus.observers import ProgressView, Speed, ElapsedRealTime, CheckPointer
from olympus.metrics.named import NamedMetric


class PPO(Task):
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
    def __init__(self, model: AbstractActorCritic, dataloader, optimizer, lr_scheduler, device,
                 ppo_epoch=5, ppo_batch_size=32, ppo_clip_param=10, ppo_max_grad_norm=1000, criterion=None,
                 storage=None, logger=None):
        super(PPO, self).__init__(device=device)

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

        self.ppo_epoch = ppo_epoch
        self.ppo_batch_size = ppo_batch_size
        self.ppo_clip_param = ppo_clip_param
        self.ppo_max_grad_norm = ppo_max_grad_norm

        self.metrics.append(NamedMetric(name='loss'))
        self.metrics.append(ElapsedRealTime())
        self.metrics.append(Speed())
        self.metrics.append(ProgressView(speed_observer=self.metrics.get('Speed')))

        if storage:
            self.metrics.append(CheckPointer(storage=storage))

        self.hyper_parameters = {}
        self.batch_size = None

    def finish(self):
        super().finish()
        if hasattr(self.dataloader, 'close'):
            self.dataloader.close()

    # Hyper Parameter Settings
    # ---------------------------------------------------------------------
    def get_space(self, **fidelities):
        """Return hyper parameter space"""
        return {
            'task': {  # fidelity(min, max, base logarithm)
                'epochs': fidelities.get('epochs')
            },
            'optimizer': self.optimizer.get_space(),
            'lr_schedule': self.lr_scheduler.get_space(),
            'model': self.actor_critic.get_space(),
            'gamma': 'loguniform(0.99, 1)'
        }

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

        self.metrics.on_new_trial(self, parameters, uid)
        self.set_device(self.device)

    def compute_returns(self, value, actions):
        reward = value
        returns = []

        for action in reversed(actions):
            reward = action.reward + self.gamma * reward * action.mask
            returns.insert(0, reward)

        return returns

    def ppo(self, current_state, replay_vector):
        """New policy gradient methods for reinforcement learning, which alternate  between  split  data
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

        self.batch_size = self.ppo_batch_size
        for p in range(self.ppo_epoch):
            sampler = BatchSampler(
                sampler=SubsetRandomSampler(range(len(state))),
                batch_size=self.ppo_batch_size,
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
                L2 = torch.clamp(ratio, 1 - self.ppo_clip_param, 1 + self.ppo_clip_param) * advantage_batch

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
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.ppo_max_grad_norm)
                self.optimizer.step()

                epoch_loss += loss.item()
                self.frame_count += self.batch_size

            epoch_loss /= count + 1
            all_loss += epoch_loss

        return {
            'loss': all_loss / self.ppo_epoch
        }

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

                context = self.ppo(last_state, vector)
                self.metrics.on_new_batch(step, self, context=context)
                step += 1

                # Called every time a simulation gets completed
                if self._epoch > prev:
                    self.metrics.on_new_epoch(self._epoch, self, None)
                    self.lr_scheduler.epoch(self._epoch)

                    prev = self._epoch
                    step = 0

                # Epochs is the number of completed simulations
                if self.dataloader.completed_simulations + 1 >= epochs:
                    break

            self.report(pprint=True, print_fun=print)

        self.finish()

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

    def _fix(self):
        # -----------------------------
        # RL Creates a lot of small torch.tensor
        # They need to be GCed so pytorch can reuse that memory
        import gc
        # Only GC the most recent gen because that where the small tensors are
        gc.collect(2)
        # -----------------------------
