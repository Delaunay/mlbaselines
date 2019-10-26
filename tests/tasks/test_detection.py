import argparse
from torch.utils.data import DataLoader
import torch.optim

import torchvision.models
import olympus.transforms as T

from olympus.tasks.detection import ObjectDetection
from olympus.datasets.detection import PennFudanDataset

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='/fast/PennFudanPed/', help='Data location')
parser.add_argument('--epochs', default=5, type=int, help='Number of epochs')
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--lr', default=0.02, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=1e-4, type=float)
parser.add_argument('--model', default='maskrcnn_resnet50_fpn', type=str)

args = parser.parse_args()



# --
dataset = PennFudanDataset(
    args.data,
    T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(0.5)
    ])
)


def collate_fn(batch):
    return tuple(zip(*batch))


num_classes = 2
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    collate_fn=collate_fn,
    shuffle=True,
    num_workers=4)

models = torchvision.models.detection.__dict__
detector = models[args.model](num_classes=num_classes, pretrained=False)

optim = torch.optim.SGD(
    detector.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay
)


def reduce_loss(loss_dict):
    return sum(loss for loss in loss_dict.values())


criterion = reduce_loss

object_detection = ObjectDetection(
    detector=detector,
    optimizer=optim,
    criterion=criterion
)

device = torch.device('cuda')
object_detection.device = device

for epoch in range(args.epochs):
    losses = []

    for nbatch, batch in enumerate(dataloader):
        images, targets = batch

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss = object_detection.fit(epoch, input=(images, targets))
        losses.append(loss)

    loss = sum([loss.item() for loss in losses]) / len(losses)

