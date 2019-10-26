from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms

import orion.client.manual

from olympus.tasks import Classification
from olympus.metrics import ValidationAccuracy
import olympus.distributed.multigpu as distributed


def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    distributed.arguments(parser)

    return parser


args = arg_parser().parse_args()

hpo_client = Orion(
    'mongodb://setepenre:1234pass@192.168.0.106:9000'
)

hpo_client.new_trial(
    experiment=f'classification_{args.model}',
    algo=dict(
        name='ASHA',
        max_trials=1000
    ),
    parameters=[
        'batch_size~uniform(1, 256)',
        'epochs~fidelity(5)',
        'lr~lognormal(1e-10, 1)',
        '',
    ]
)

# Apply Orion overrides
args = hpo_client.sample(args)
distributed.enable_distributed_process(args)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


val_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '/tmp/data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    ),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=1,
    pin_memory=True)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '/tmp/data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    ),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=1,
    pin_memory=True)

model_ = Net()
model_ = hpo_client.resume(model_)
model = distributed.data_parallel(model_)

task = Classification(
    classifier=model,
    optimizer=optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum)
)
task.device = torch.device('cpu')

task.metrics.append(
    ValidationAccuracy(loader=val_loader)
)

for i in range(args.epochs):
    for batch_idx, input in enumerate(train_loader):
        if batch_idx == 3:
            break

        task.fit(batch_idx, input, None)

# push the latest metrics
task.finish()

task.report(pprint=True, print_fun=print)

hpo_client.report(
    name='ValidationAccuracy',
    type='objective',
    value=task.metrics()['ValidationAccuracy']
)
