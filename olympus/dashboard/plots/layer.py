import torch
import torch.nn as nn
from torch.autograd import Variable

from olympus.optimizers import Optimizer
from olympus.utils import debug, fetch_device


class LayerViz:
    """Find the image the maximize the activation of a given layer"""

    def __init__(self, model, layer, index):
        self.model = model
        self.layer = layer
        self.index = index
        self._loss = None
        self.device = fetch_device()

    @property
    def loss(self):
        return self._loss.item()

    def execute_model(self, layers, image):
        x = image

        for i, layer in enumerate(layers):
            x = layer(x)

            if layer is self.layer:
                break

        return x

    def __call__(self, image=None, steps=32, return_grad=False, lr=1):
        """Find the image the maximize the activation of a given layer

        Parameters
        ----------
        image: Tensor
            Image that will be used to start the optimization
            If not provided a random image will be generated

        steps: int
            Number of optimization steps to do

        return_grad: bool
            Returns the accumulated grad instead of the image

        lr: float
            Learning rate for the optimizer

        Returns
        -------
        The modified image that optimize a given layer
        """
        self.model.eval()
        self.model = self.model.to(device=self.device)

        if image is None:
            image = torch.randn(1, 3, 224, 224)

        image = image.to(device=self.device)

        layers = list([module for module in self.model.modules() if type(module) != nn.Sequential])
        skip = int(isinstance(layers[0], type(self.model)))
        layers = layers[skip:]

        image = Variable(image.to(device=self.device), requires_grad=True)
        optimizer = Optimizer(
            'sgd',
            params=[image],
            lr=lr,
            weight_decay=1e-6,
            momentum=0)

        grad = None
        for i in range(steps):
            optimizer.zero_grad()
            x = self.execute_model(layers, image)

            loss = - torch.mean(x[:, self.index])
            debug(f'{i:4d} (loss: {loss.item():.4f})')

            optimizer.backward(loss)
            optimizer.step()
            self._loss = loss.detach()

            if return_grad and grad is None:
                grad = image.grad.detach()

            elif return_grad:
                grad += image.grad

        if return_grad:
            return (grad - grad.min()) / grad.max()

        return ((image - image.min()) / image.max()).cpu()


def find_good_layer(model, image, top=16, return_rank=False):
    """Try to find the the layer/filter that explain the image the best,
    Prints top 16 (layer, channel) combination

    Parameters
    ----------
    model:
        Pytorch model

    image: Tensor
        Image to use

    top: int
        Number of configuration to print

    return_rank: bool
        return the full ranking of all the configuration that was tried

    Examples
    --------

    .. code-block:: python

        import torchvision.models as models
        from olympus.dashboard.plots.saliency import imagenet_postprocessor
        from PIL import Image

        model = models.vgg19(pretrained=True)

        path = '/home/setepenre/work/olympus/docs/_static/images/cat.jpg'
        img = Image.open(path)

        img = find_good_layer(model, image=torch.randn(1, 3, 224, 224))
        imagenet_postprocessor(img)[0].save('filter.jpg')


    .. image:: ../../../docs/_static/images/cat_conv_filter_mosaic.jpg

    """
    layers = list([module for module in model.modules() if type(module) != nn.Sequential])
    conv_layers = list(filter(lambda l: isinstance(l, nn.Conv2d), layers))

    costs = []
    current = dict(loss=10000)

    for i, conv in enumerate(conv_layers):
        channels = conv.out_channels

        for c in range(channels):
            try:
                viz = LayerViz(model, layer=conv, index=c)
                viz(steps=4, image=image)

                config = dict(layer=conv, channel=c, loss=viz.loss, layer_index=i)
                costs.append(config)

                if viz.loss < current['loss']:
                    current = config
                    print(config)

            except Exception as e:
                print(f'Skipping {conv} because of exception {e}')

        print(f'{i * 100 / len(conv_layers):8.2f}')

    costs = sorted(costs, key=lambda elem: elem['loss'])

    if return_rank:
        return costs

    for i in range(top):
        print(costs[i])

    best = costs[0]
    viz = LayerViz(model, layer=best['layer'], index=best['channel'])
    return viz(steps=32, image=image)


def make_image_mosaic(images):
    """Make an image mosaic from a list of tensors"""
    c, width, height = images[0][0].shape
    tensor = torch.randn(c, width * len(images), height * len(images[0]))

    rstart = 0
    cstart = 0

    for row in images:
        for img in row:
            tensor[:, rstart:rstart + width, cstart:cstart + height] = img
            cstart += height

        cstart = 0
        rstart += width

    return tensor
