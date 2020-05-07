import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models._utils import IntermediateLayerGetter

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone = IntermediateLayerGetter(self.backbone, return_layers={'layer4': 'layer4', 'layer3':'layer3'})
        self.layer4_classifier = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 21, 1))
        self.layer3_classifier = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 21, 1))

    def forward(self, input):
        output = self.backbone(input)
        layer3, layer4 = output['layer3'], output['layer4']

        output = self.layer4_classifier(layer4)
        output = F.interpolate(output, size=layer3.size()[-2:], mode='bilinear', align_corners=False)
        output = self.layer3_classifier(layer3) + output
        output = F.interpolate(output, size=input.size()[-2:], mode='bilinear', align_corners=False)
        return output

    def initialize(self):
        self.layer4_classifier.apply(init_weights)
        self.layer3_classifier.apply(init_weights)

def fcn_resnet18(input_size=None, output_size=None):
    return Model()

builders = {
    'fcn_resnet18': fcn_resnet18,
}
