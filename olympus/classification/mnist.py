from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data.sampler import RandomSampler

from torchvision import datasets, transforms

from olympus.classification.trainer import TrainClassifier
from olympus.utils.sampler import ResumableSampler


def distributed_arguments(parents=None):
    if parents is None:
        parents = []

    parser = argparse.ArgumentParser(description='Distributed Arguments', parents=parents)

    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--world-size', type=int)

    return parser


def model_arguments(parents=None):
    if parents is None:
        parents = []

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example', parents=parents)

    # System
    # ------
    parser.add_argument('--cuda', dest='cuda', action='store_true', default=True,
                        help='enables CUDA training')

    parser.add_argument('--no-cuda', dest='cuda', action='store_false', default=False,
                        help='disables CUDA training')

    # Test Parameters
    # ---------------
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')

    # Training Parameters
    # -------------------
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Hyper parameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')

    # Optimizers
    # ----------
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--momentum', type=float, default=0.001, metavar='M',
                        help='SGD momentum (default: 0.9)')

    return parser


def parse_arguments(parser, argv=None, show=True):
    args = parser.parse_args(argv)

    if show:
        for k, v in vars(args).items():
            print(f'{k:>20}: {v}')

    return args


# Model
# -----
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def mnist(argv=None):
    args = parse_arguments(argv)
    use_cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Train
    # -----
    train_source = datasets.MNIST(
        '../data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
    )
    train_sampler = RandomSampler(train_source)
    train_loader = torch.utils.data.DataLoader(
        train_source,
        batch_size=args.batch_size,
        shuffle=True,
        # sampler=train_sampler,
        **kwargs
    )

    # Test
    # ----
    test_source = datasets.MNIST(
        '../data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    test_loader = torch.utils.data.DataLoader(
        test_source,
        batch_size=args.test_batch_size,
        shuffle=False,
        sampler=RandomSampler(test_source),
        **kwargs
    )

    model = Net().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum
    )

    trainer = TrainClassifier(
        optimizer,
        nn.NLLLoss(reduction='sum'),
        model,
        train_sampler,
        device)

    trainer.fit(args.epochs, train_loader)
    result = trainer.eval_model(test_loader)

    print(f'Eval (acc: {result.acc * 100}) (loss: {result.loss})')


if __name__ == '__main__':
    from mlbaselines.distributed import init_process_group

    parser = distributed_arguments()
    args = parse_arguments(parser)

    # Manually set the device ids.
    # device = args.rank % torch.cuda.device_count()
    # if args.local_rank is not None:
    #     device = args.local_rank
    # torch.cuda.set_device(device)

    init_process_group(
        backend='mongodb',
        world_size=args.world_size, rank=args.local_rank)

    mnist()
