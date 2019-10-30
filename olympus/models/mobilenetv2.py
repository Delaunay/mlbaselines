"""MobileNetV2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.

Note: Taken from https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenetv2.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    def __init__(self, cfg, input_size, conv, avgpool, num_classes=10):
        super(MobileNetV2, self).__init__()

        self.cfg = cfg

        self.conv1 = nn.Conv2d(input_size[0], 32, **conv, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.avgpool = nn.AvgPool2d(**avgpool)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def build(input_size, output_size):

    cfg = [[1,  16, 1, 1],
           [6,  24, 2, 1],
           [6,  32, 3, 2],
           [6,  64, 4, 2],
           [6,  96, 3, 1],
           [6, 160, 3, 2],
           [6, 320, 1, 1]]

    if input_size == (1, 28, 28):
        log.info('Using MobileNetV2 architecture for MNIST')
        conv = {}
        avgpool = {}
    elif input_size == (3, 32, 32):
        log.info('Using MobileNetV2 architecture for CIFAR10/100')
        conv = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        avgpool = {'kernel_size': 4}
    elif input_size == (3, 64, 64):
        log.info('Using MobileNetV2 architecture for TinyImageNet')
        conv = {'kernel_size': 3, 'stride': 2, 'padding': 1}
        avgpool = {'kernel_size': 7}
        cfg[1][-1] = 2
    # TODO: Add support for ImageNet

    return MobileNetV2(cfg, input_size, num_classes=output_size, conv=conv, avgpool=avgpool)


builders = {
    'mobilenetv2': build}
