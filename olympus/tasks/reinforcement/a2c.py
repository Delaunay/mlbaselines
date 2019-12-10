from typing import Callable

import numpy as np
import torch
from torch.distributions import Categorical, Distribution
from torch.nn import Module
from torch.optim import Optimizer

from olympus.tasks.task import Task
from olympus.utils import info, select
from olympus.reinforcement.replay import ReplayVector
from olympus.reinforcement.utils import AbstractActorCritic, SavedAction
from olympus.observers import ProgressView, Speed, ElapsedRealTime, CheckPointer, Tracker
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
    def __init__(self, model: AbstractActorCritic, dataloader, optimizer, storage, lr_scheduler, device, logger, gamma=0.99, num_steps=5, criterion=None):
        super(A2C, self).__init__(device=device)

        if criterion is None:
            criterion = lambda x: x.sum()

        self.actor_critic = model
        self.lr_scheduler = lr_scheduler
        self.optimizer: Optimizer = optimizer
        self.criterion: Module = criterion
        self.num_steps: int = num_steps
        self.gamma: float = gamma
        self.eps = np.finfo(np.float32).eps.item()
        self.action_sampler: Callable[[], Distribution] = Categorical
        self.tensor_shape = None
        self.frame_count: int = 0
        self.dataloader = dataloader
        self.storage = storage
        self._first_epoch = 0

        self.metrics.append(NamedMetric(name='loss'))
        self.metrics.append(ElapsedRealTime())
        self.metrics.append(Speed())
        self.metrics.append(ProgressView(speed_observer=self.metrics.get('Speed')))

        if storage:
            self.metrics.append(CheckPointer(storage=storage))

        if logger is not None:
            self.metrics.append(Tracker(logger=logger))

        self.hyper_parameters = {}
        self.batch_size = None

    def finish(self):
        super().finish()
        if hasattr(self.dataloader, 'close'):
            self.dataloader.close()

    # Hyper Parameter Settings
    # ---------------------------------------------------------------------
    def init(self, optimizer=None, lr_schedule=None, model=None, trial_id=None):
        """
        Parameters
        ----------
        optimizer: Dict
            Optimizer hyper parameters

        lr_schedule: Dict
            lr schedule hyper parameters

        model: Dict
            model hyper parameters

        trial_id: Optional[str]
            trial id to use for logging.
            When using orion usually it already created a trial for us we just need to append to it
        """

        optimizer = select(optimizer, {})
        lr_schedule = select(lr_schedule, {})
        model = select(model, {})

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

        # try to resume itself
        checkpoints = self.metrics.get('CheckPointer')
        if checkpoints is not None:
            self.resume(checkpoints.storage)

        parameters = {}
        parameters.update(optimizer)
        parameters.update(lr_schedule)
        parameters.update(model)

        self.metrics.on_new_trial(self, parameters, trial_id)
        self.set_device(self.device)

    def get_space(self, **fidelities):
        """Return hyper parameter space"""
        return {
            'task': {       # fidelity(min, max, base logarithm)
                'epochs': fidelities.get('epochs')
            },
            'optimizer': self.optimizer.get_space(),
            'lr_schedule': self.lr_scheduler.get_space(),
            'model': self.actor_critic.get_space()
        }

    # Training
    # --------------------------------------------------------------------
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

    def fit(self, epochs, context=None):
        # -----------------------------
        # RL Creates a lot of small torch.tensor
        # They need to be GCed so pytorch can reuse that memory
        import gc
        # Only GC the most recent gen because that where the small tensors are
        gc.collect(2)
        # -----------------------------
        self.actor_stream = torch.cuda.streams.Stream()
        self.critic_stream = torch.cuda.streams.Stream()

        self._start(epochs)

        for epoch in range(0, epochs):
            frame_count = self.frame_count
            self.epoch(epoch + 1, self.dataloader.iterator())
            frame_count = self.frame_count - frame_count

            self.metrics.on_new_epoch(epoch + 1, self, context)
            self.lr_scheduler.epoch(epoch)

        self.report(pprint=True, print_fun=print)

    def epoch(self, epoch, iterator):
        # starting epoch
        state = iterator.next(action=None)
        state = state.to(self.device)

        is_done = False
        batch = 0

        while not is_done:
            batch, is_done = self.step(batch, state, iterator)

        return

    def step(self, batch, state, iterator):
        # Make sure we have as much memory as possible
        replay_vector = ReplayVector()
        is_done = False

        # Accumulates simulation steps in batches
        for step in range(self.num_steps):
            # Run the actor and critic in parallel
            with torch.cuda.stream(self.actor_stream):
                action, log_prob, entropy = self.actor_critic.act(state)

            with torch.cuda.stream(self.critic_stream):
                critic = self.actor_critic.critic(state)

            self.actor_stream.synchronize()
            new_state = iterator.next(action.detach())
            self.critic_stream.synchronize()

            if new_state is not None:
                old_state = state
                state, reward, done, _ = new_state
                state = state.to(self.device)

                # Make sure all Tensor follows the size below
                #   (WorkerSize x size) using unsqueeze
                replay_vector.append(SavedAction(
                    action=action,
                    reward=reward.to(device=self.device),
                    log_prob=log_prob.squeeze(1),
                    entropy=entropy,
                    critic=critic,
                    mask=(1 - done).to(device=self.device),
                    state=old_state,
                    next_state=state
                ))
            else:
                is_done = True
                break

        # optimize for current batch
        if replay_vector:
            batch += 1
            results = self.advantage_actor_critic(state, replay_vector)

            self.metrics.on_new_batch(batch, self, None, results)
            self.lr_scheduler.step(batch)

        return batch, is_done

    # CheckPointing
    # ---------------------------------------------------------------------
    def resume(self, storage):
        state_dict = storage.safe_load('checkpoint', device=self.device)

        if not state_dict:
            info('Starting from scratch')
            return False

        try:
            self._first_epoch = state_dict['epoch']
            info(f"Resuming from (epoch: {self._first_epoch})")

            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'], device=self.device)
            self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
            self.dataloader.load_state_dict(state_dict['dataloader'])
            self.metrics.load_state_dict(state_dict['metrics'])

            return True
        except KeyError as e:
            raise KeyError(f'Bad state dictionary!, missing (key: {e.args}) in {self.storage.folder}') from e

    def checkpoint(self, epoch, storage):
        info(f'Saving checkpoint (epoch: {epoch})')
        was_saved = storage.save(
            'checkpoint',
            dict(
                epoch=epoch,
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                lr_scheduler=self.lr_scheduler.state_dict(),
                dataloader=self.dataloader.state_dict(),
                metrics=self.metrics.state_dict()
            )
        )
        if was_saved:
            info('Checkpoint saved')
        else:
            info('Skipped Checkpoint')

    @property
    def model(self):
        return self.actor_critic

    @model.setter
    def model(self, model):
        self.actor_critic = model
