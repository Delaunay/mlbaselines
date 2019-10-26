from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from olympus.tasks import GAN
from olympus.metrics import ValidationAccuracy


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

    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    return parser


args = arg_parser().parse_args()

nz = int(args.nz)
ngf = int(args.ngf)
ndf = int(args.ndf)
nc = 1


class Generator(nn.Module):
    def __init__(self, input_size, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.check(input_size)

    def check(self, input_size):
        tensor = torch.randn((1,) + input_size)
        return self.main(tensor)

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, input_size, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.check(input_size)

    def check(self, input_size, print_fun=None, verbose=False):
        print_buffer = []
        if print_fun is None:
            print_fun = lambda *pargs, **kwargs: print_buffer.append((pargs, kwargs))

        def print_diag():
            for p, k in print_buffer:
                print(*p, **k)

        tensor = torch.randn((2,) + input_size)

        try:
            for idx, i in enumerate(self.main.children()):
                print_fun(idx + 1, i, '\n\t', tensor.shape, end='\t => ')
                tensor  = i(tensor)
                print_fun(tensor.shape)
        except:
            print_diag()
            raise

        if verbose:
            print_diag()

        return tensor

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


val_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '/tmp/data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize((64, 64)),
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
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    ),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=1,
    pin_memory=True)


image_size = (1, 64, 64)
latent_vector = (nz, 1, 1)

gen = Generator(latent_vector, nz, ngf, nc)
disc = Discriminator(image_size, nc, ndf)

task = GAN(
    generator=gen,
    discriminator=disc,
    criterion=nn.BCELoss(),
    generator_optimizer=optim.Adam(gen.parameters()),
    discriminator_optimizer=optim.Adam(disc.parameters()),
    latent_vector_size=nz
)
task.device = torch.device('cpu')


for i in range(args.epochs):
    for batch_idx, input in enumerate(train_loader):
        if batch_idx == 3:
            break

        task.fit(batch_idx, input, None)

# push the latest metrics
task.finish()

task.report(pprint=True, print_fun=print)
