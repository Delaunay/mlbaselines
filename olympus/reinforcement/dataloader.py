import torch
import numpy as np

from olympus.reinforcement.replay import ReplayVector
from olympus.reinforcement.utils import SavedAction
from olympus.utils.cuda import Stream, stream


def to_nchw(states):
    return states.permute(0, 3, 1, 2)


class RLTorchIterator:
    """Iterates through environment states

    Parameters
    ----------
    actor: Union[nn.Module, Callable]
        Returns the action that should be taken

    critic: Union[nn.Module, Callable]
        Returns the value of the current state

    max_step: Optional[int]
        If unspecified the Iterator is infinite, else we stop after max_steps

    no_grad: bool
        Whether or not the actor and the critic should have their grad computed

    Returns
    -------
    A dictionary representing the transition from one state to anoter

    state: Tensor[NCHW, dtype=uint8]
        State of the game before the action is taken for images (size: (num_parallel, 3, H, W))

    new_state: Tensor[NCHW, dtype=uint8]
        State of the game after the action is taken for images (size: (num_parallel, 3, H, W))

    action: Tensor[num_parallel, dtype=int]
        Return the action taken for each parallel simulation

    log_prob: Tensor[num_parallel, dtype=float]
    entropy: Tensor[num_parallel, dtype=float]
    critic: Tensor[num_parallel, dtype=float]
    reward: Tensor[num_parallel, dtype=float]
    done: Tensor[num_parallel, dtype=bool]
    info: List[dict] size: num_parallel
    """
    def __init__(self, environment, actor, critic, device=None, max_step=None, no_grad=False):
        self.step = 0
        self.max_step = max_step
        self.env = environment
        self.actor = actor
        self.critic = critic
        self.actor_stream = Stream()
        self.critic_stream = Stream()
        self.grad_ctx = torch.enable_grad
        self.dtype = torch.float
        self.completed_simulations = 0
        self.device = device
        if self.device is None:
            self.device = torch.device('cpu')

        if no_grad:
            self.grad_ctx = torch.no_grad

        self.state = self._convert(self.env.reset()).to(device=self.device)

    def to(self, device):
        self.device = device
        self.state = self.state.to(device=self.device)
        return self

    def close(self):
        return self.env.close()

    def __iter__(self):
        return self

    def __len__(self):
        return 1

    def _convert(self, x):
        if x is None:
            return None

        return torch.from_numpy(
            np.stack(x)
        ).to(dtype=self.dtype, device=self.device)

    def __next__(self):
        if self.max_step is not None and self.step >= self.max_step:
            raise StopIteration

        with self.grad_ctx():
            with stream(self.actor_stream):
                action, log_prob, entropy = self.actor(self.state)

            critic = None
            if self.critic:
                with stream(self.critic_stream):
                    critic = self.critic(self.state)

        self.actor_stream.synchronize()
        state, rew, done, info = self.env.step(action.cpu().numpy())
        state = self._convert(state).to(device=self.device)
        self.critic_stream.synchronize()

        transition = {
            'state'    : self.state,            # Tensor[NCHW, uint8]
            'new_state': state,                 # Tensor[NCHW, uint8]
            'action'   : action,                # List[N, int]
            'log_prob' : log_prob.squeeze(1),   # List[N, int]
            'entropy'  : entropy,               # List[N, int]
            'critic'   : critic,                # List[N, int]
            'reward'   : self._convert(rew),    # List[N, Float]
            'done'     : self._convert(done),   # List[N, Bool]
            'info'     : info                   # List[N, dict]
        }

        self.completed_simulations += transition.get('done').sum()
        self.step += 1
        self.state = state
        return transition


class ReplayVectorIterator:
    """Aggregate Transition into a vector to be used for later"""
    def __init__(self, iterator: RLTorchIterator, num_steps):
        self.iterator = iterator
        self.num_steps = num_steps

    def to(self, device):
        self.iterator.to(device)
        return self

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterator)

    def __next__(self):
        replay = ReplayVector()

        for step_idx, transition in enumerate(self.iterator):
            replay.append(SavedAction(
                action=transition.get('action'),
                reward=transition.get('reward'),
                log_prob=transition.get('log_prob'),
                entropy=transition.get('entropy'),
                critic=transition.get('critic'),
                mask=(1 - transition.get('done')),
                state=transition.get('state'),
                next_state=transition.get('new_state'),
                info=transition.get('info')
            ))

            if step_idx + 1 >= self.num_steps:
                break

        return replay

    def close(self):
        return self.iterator.close()

    @property
    def state(self):
        """Return the latest state"""
        return self.iterator.state

    @property
    def completed_simulations(self):
        """Number of completed simulations since start"""
        return self.iterator.completed_simulations


def simple_replay_vector(num_steps):
    def _replay(iterator):
        return ReplayVectorIterator(iterator, num_steps)

    return _replay


class RLDataLoader:
    """
    Parameters
    ----------
    dataset_environment:
        Generic Reinforcement Learning environment

    replay:
        Replay Vector iterator constructor

    transform:
        Transform to apply to each simulation state
    """
    def __init__(self, dataset_environment, actor, critic, replay=None):
        self.dataset = dataset_environment
        self.replay = replay
        self.actor = actor
        self.critic = critic
        self.device = torch.device('cpu')

        if self.replay is None:
            self.replay = simple_replay_vector(1)

    def train(self, no_grad=False):
        return self.replay(RLTorchIterator(
            self.dataset.train,
            self.actor,
            self.critic,
            device=self.device,
            no_grad=no_grad))

    def valid(self):
        return self.replay(RLTorchIterator(
            self.dataset.valid,
            self.actor,
            self.critic,
            device=self.device,
            no_grad=True))

    def test(self):
        return self.replay(RLTorchIterator(
            self.dataset.test,
            self.actor,
            self.critic,
            device=self.device,
            no_grad=True))

    def shutdown(self):
        self.dataset.close()

    def close(self):
        self.shutdown()


if __name__ == '__main__':
    from olympus.reinforcement.procgenenv import ProcgenEnvironment

    env = ProcgenEnvironment('coinrun', parallel_env=4)

    def dummy_actor(*args, **kwargs):
        return env.sample_action(), [0, 0, 0, 0], [0, 0, 0, 0]

    loader = RLDataLoader(
        env,
        replay=simple_replay_vector(num_steps=2),
        actor=dummy_actor,
        critic=lambda x: [0, 0, 0, 0]
    )

    train_set = loader.train()
    for step, i in enumerate(train_set):
        for k, v in i.to_dict().items():

            if isinstance(v, torch.Tensor):
                print(f'{k:>30} :', v.shape, v.dtype)
            else:
                print(f'{k:>30} :', v)

        break
