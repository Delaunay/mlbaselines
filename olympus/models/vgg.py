import logging
import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


log = logging.getLogger(__name__)

# Possible configuration variants of VGG network
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(self, layers, input_size, num_classes, batch_norm):
        super(VGG, self).__init__()

        if input_size == (1, 28, 28):
            log.info('Using VGG architecture for MNIST')
            classifier = {'input': 512, 'hidden': None}
        elif input_size == (3, 32, 32):
            log.info('Using VGG architecture for CIFAR10/100')
            classifier = {'input': 512, 'hidden': None}
        elif input_size == (3, 64, 64):
            log.info('Using VGG architecture for TinyImageNet')
            classifier = {'input': 2048, 'hidden': 1024}
        # TODO: Add support for ImageNet
        else:
            raise ValueError(
                'There is no VGG architecture for an input size {}'.format(input_size))

        self.features = self.make_layers(input_size[0], layers, batch_norm)
        
        if classifier.get('hidden'):
            self.classifier = nn.Sequential(
                nn.Linear(classifier['input'], classifier['hidden']),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(classifier['hidden'], classifier['hidden']),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(classifier['hidden'], num_classes),
            )
        else:
            self.classifier = nn.Linear(classifier['input'], num_classes)
            
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def make_layers(self, in_channels, cfg, batch_norm):
        layers = []
        
        for v in cfg:
            if isinstance(v, str) and v.startswith('M'):
                layers += [nn.MaxPool2d(kernel_size=len(v) + 1, stride=len(v) + 1)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                # con2d.register_forward_hook(save_computations)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
        
        
    def save_computations(self, input, output):
        setattr(self, "input", input)
        setattr(self, "output", output)


def distribute(model, distributed):
    # AlexNet and VGG should be treated differently
    #         # DataParallel will divide and allocate batch_size to all available GPUs
    #         if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #             model.features = torch.nn.DataParallel(model.features)
    #             model.cuda()
    #         else:
    #             model = torch.nn.DataParallel(model).cuda()

    # Because last fc layer is big and not suitable for DataParallel
    # Source: https://github.com/pytorch/examples/issues/144

    if distributed > 1:
        if distributed != torch.cuda.device_count():
            raise RuntimeError("{} GPUs are required by the configuration but {} are currently "
                               "made available to the process.".format(
                                   distributed, torch.cuda.device_count()))

        if isinstance(model.features, nn.Sequential):
            model.features = torch.nn.DataParallel(model.features).cuda()
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    return model


def build_vgg11(input_size, output_size):
    return VGG(cfg['vgg11'], input_size=input_size, num_classes=output_size, batch_norm=True)


def build_vgg13(input_size, output_size):
    return VGG(cfg['vgg13'], input_size=input_size, num_classes=output_size, batch_norm=True)


def build_vgg16(input_size, output_size):
    return VGG(cfg['vgg16'], input_size=input_size, num_classes=output_size, batch_norm=True)


def build_vgg19(input_size, output_size):
    return VGG(cfg['vgg19'], input_size=input_size, num_classes=output_size, batch_norm=True)


builders = {
    'vgg11': build_vgg11,
    'vgg13': build_vgg13,
    'vgg16': build_vgg16,
    'vgg19': build_vgg19}
