import functools

import numpy

from torchvision.models import detection


def maskrcnn_resnet50_fpn(input_size=None, output_size=None):
    """with pretrained_backbone"""
    if not isinstance(output_size, int):
        output_size = numpy.product(input_size)

    return detection.maskrcnn_resnet50_fpn(num_classes=output_size)


def keypointrcnn_resnet50_fpn(input_size=None, output_size=None):
    """with pretrained_backbone"""
    if not isinstance(output_size, int):
        output_size = numpy.product(input_size)

    return detection.keypointrcnn_resnet50_fpn(num_classes=output_size)


def _fasterrcnn_resnet_fpn(backbone='resnet50', num_classes=91, pretrained_backbone=True, **kwargs):
    from torchvision.models.detection.faster_rcnn import FasterRCNN, resnet_fpn_backbone

    backbone = resnet_fpn_backbone(backbone, pretrained_backbone)
    model = FasterRCNN(backbone, num_classes, **kwargs)

    detection.fasterrcnn_resnet50_fpn()
    return model


def fasterrcnn_resnet_fpn(input_size=None, output_size=None, backbone='resnet50'):
    """with pretrained_backbone"""
    if not isinstance(output_size, int):
        output_size = numpy.product(input_size)

    return _fasterrcnn_resnet_fpn(num_classes=output_size, backbone=backbone)


builders = {
    'maskrcnn_resnet50_fpn': maskrcnn_resnet50_fpn,
    'keypointrcnn_resnet50_fpn': keypointrcnn_resnet50_fpn,
    'fasterrcnn_resnet50_fpn': functools.partial(fasterrcnn_resnet_fpn, backbone='resnet50'),
    'fasterrcnn_resnet18_fpn': functools.partial(fasterrcnn_resnet_fpn, backbone='resnet18')
}


