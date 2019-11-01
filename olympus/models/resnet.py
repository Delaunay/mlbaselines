import functools
from olympus.utils import info
import torch.nn as nn


# All models here are assumed to accept RGB input, thus 3 input channels.
# Checkpoints of models pre-trained on Imagenet.
# NOT SUPPORTED YET
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, input_size, conv, maxpool, avgpool, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size[0], 64, **conv, bias=False)
        # For ImageNet
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        if maxpool:
            # For ImageNet
            # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.maxpool = nn.MaxPool2d(**maxpool)
        else:
            self.maxpool = None

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if avgpool:
            self.avgpool = nn.AvgPool2d(**avgpool)
        else:
            self.avgpool = None

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.maxpool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.avgpool:
            x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def build(block, cfg, input_size, output_size):
    if input_size == (1, 28, 28):
        info('Using PreActResNet architecture for MNIST')
        conv = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        avgpool = {'kernel_size': 4}
        maxpool = {}
    elif input_size == (3, 32, 32):
        info('Using PreActResNet architecture for CIFAR10/100')

        conv = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        avgpool = {'kernel_size': 4}
        maxpool = {}
    elif input_size == (3, 64, 64):
        info('Using PreActResNet architecture for TinyImageNet')

        conv = {'kernel_size': 7, 'stride': 2, 'padding': 3}
        avgpool = {'kernel_size': 2}
        maxpool = {'kernel_size': 3, 'stride': 2, 'padding': 1}

    model = ResNet(
        block,
        cfg,
        input_size=input_size,
        conv=conv,
        maxpool=maxpool,
        avgpool=avgpool,
        num_classes=output_size
    )

    return model


builders = {
    'resnet18': functools.partial(build, block=BasicBlock, cfg=[2, 2, 2, 2]),
    'resnet34': functools.partial(build, block=BasicBlock, cfg=[3, 4, 6, 3]),
    'resnet50': functools.partial(build, block=Bottleneck, cfg=[3, 4, 6, 3]),
    'resnet101': functools.partial(build, block=Bottleneck, cfg=[3, 4, 23, 3]),
    'resnet152': functools.partial(build, block=Bottleneck, cfg=[3, 8, 36, 3])
}
