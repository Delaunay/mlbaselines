"""Pre-activation ResNet in PyTorch"""

# .. rubric:: References
#
# .. [PreactResnet] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun:
#     Identity Mappings in Deep Residual Networks. arXiv:1603.05027

import functools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


log = logging.getLogger(__name__)


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, first=False):
        super(PreActBlock, self).__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        if not self.first:
            out = F.relu(self.bn1(x))
        else:
            out = x
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, first=False):
        super(PreActBottleneck, self).__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        if not self.first:
            out = F.relu(self.bn1(x))
        else:
            out = x
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, input_size, conv, maxpool, avgpool,  num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_size[0], 64, **conv, bias=False)

        if maxpool:
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(**maxpool)
            first = True
        else:
            self.maxpool = None
            first = False

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,
                                       first=first)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512*block.expansion)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        if avgpool:
            self.avgpool = nn.AvgPool2d(**avgpool)
        else:
            self.avgpool = None

        # TODO
        ## Zero-initialize the last BN in each residual branch,
        ## so that the residual branch starts with zeros, and each residual block behaves like an identity.
        ## This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        #if True:  # zero_init_residual:
        #    for m in self.modules():
        #        if isinstance(m, PreActBottleneck):
        #            nn.init.constant_(m.bn3.weight, 0)
        #        elif isinstance(m, PreActBlock):
        #            nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, num_blocks, stride, first=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, first))
            self.in_planes = planes * block.expansion
            first = False
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.maxpool is not None:
            out = self.maxpool(F.relu(self.bn1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        if self.avgpool is not None:
            out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def build(block, cfg, input_size, output_size):

    if input_size == (1, 28, 28):
        log.info('Using PreActResNet architecture for MNIST')
        conv = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        avgpool = {'kernel_size': 4}
        maxpool = {}
    elif input_size == (3, 32, 32):
        log.info('Using PreActResNet architecture for CIFAR10/100')
        conv = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        avgpool = {'kernel_size': 4}
        maxpool = {}
    elif input_size == (3, 64, 64):
        log.info('Using PreActResNet architecture for TinyImageNet')
        conv = {'kernel_size': 7, 'stride': 2, 'padding': 3}
        avgpool = {'kernel_size': 2}
        maxpool = {'kernel_size': 3, 'stride': 2, 'padding': 1}

    return PreActResNet(block, cfg, input_size=input_size, num_classes=output_size, conv=conv,
                        maxpool=maxpool, avgpool=avgpool)


builders = {
    'preactresnet18': functools.partial(build, block=PreActBlock, cfg=[2, 2, 2, 2]),
    'preactresnet34': functools.partial(build, block=PreActBlock, cfg=[3, 4, 6, 3]),
    'preactresnet50': functools.partial(build, block=PreActBottleneck, cfg=[3, 4, 6, 3]),
    'preactresnet101': functools.partial(build, block=PreActBottleneck, cfg=[3, 4, 23, 3]),
    'preactresnet152': functools.partial(build, block=PreActBottleneck, cfg=[3, 8, 36, 3])
    }
