import torch
import torch.nn as nn
import torchvision.transforms as transforms
from olympus.transforms import Denormalize


def imagenet_preprocess(img):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    preprocessor = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize
    ])

    if isinstance(img, list):
        return torch.stack([preprocessor(i) for i in img])

    return torch.stack([preprocessor(img)])


def imagenet_postprocessor(image):
    denorm = Denormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    def renorm(img):
        img -= img.min()
        img /= img.max()
        return img

    post = transforms.Compose([
        denorm,
        renorm,
        transforms.ToPILImage(),
        # transforms.Grayscale(),
    ])

    if len(image.shape) == 4:
        results = []
        for i in range(image.shape[0]):
            img = image[i]
            results.append(post(img))

    else:
        return post(image)

    return results


class GuidedBackprop:
    """
    TODO: find original paper

    Parameters
    ----------
    model:
        Pytorch model

    activation:
        Type of the activation layer to use, defaults to ReLU

    preprocessor: Callable[[List[PILImage]], Tensor]
        use to apply preprocessing to images

    postprocesoor: Callable[[Tensor], List[PilImage]]
        used to reconstruct the image from a tensor

    References
    ----------
    .. [1] J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller.
        "Striving for Simplicity: The All Convolutional Net"
        https://arxiv.org/abs/1412.6806

    .. [2] K. Simonyan, A. Vedaldi, A. Zisserman.
        "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps"
        https://arxiv.org/abs/1312.6034

    .. [3] https://arxiv.org/pdf/1810.03292v1.pdf

    Examples
    --------

    >>> import torchvision.models as models
    >>> from torchvision import transforms
    >>> from PIL import Image

    >>> path = 'docs/_static/images/cat.jpg'
    >>> img = Image.open(path)

    >>> model = models.alexnet(pretrained=True)

    >>> guided = GuidedBackprop(model)
    >>> _ = guided([img], [285])

    >>> for i, grad in enumerate(guided.negative_saliency()):
    ...     img = imagenet_postprocessor(grad)
    ...     img.save(f'negative_saliency_{i}.jpg')

    .. image:: ../../../docs/_static/images/cat.jpg
        :width: 45 %

    .. image:: ../../../docs/_static/images/negative_saliency.jpg
        :width: 45 %

    """
    def __init__(self, model, activation=nn.ReLU, preprocessor=imagenet_preprocess, postprocessor=imagenet_postprocessor):
        self.gradients = None
        self.inputs = None

        self.activation_stack = []
        self.model = model
        self.activation = activation

        self._register_hooks(model, activation)
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def _register_hooks(self, model, activation=nn.ReLU):
        layers = list([module for module in model.modules() if type(module) != nn.Sequential])

        # first layer is self
        skip = int(isinstance(layers[0], type(model)))
        layers = layers[skip:]

        first_layer = layers[0]
        first_layer.register_backward_hook(self.fetch_gradient)

        # Hook to activation
        for module in layers:
            if isinstance(module, activation):
                module.register_backward_hook(self.activation_backward)
                module.register_forward_hook(self.activation_forward)

        return self

    def fetch_gradient(self, module, grad_in, grad_out):
        """Fetch last gradient or gradient of the first layer"""
        self.gradients = grad_in[0].detach()

    def activation_forward(self, module, ten_in, ten_out):
        """Forward hook to the activation layer"""
        self.activation_stack.append(ten_out.detach())

    def activation_backward(self, module, grad_in, grad_out):
        """Backward hook to the activation layer"""
        output = self.activation_stack.pop()
        output[output > 0] = 1

        new_grad = output * torch.clamp(grad_in[0], min=0.0)
        return new_grad,

    def positive_saliency(self):
        """Returns positive gradients"""
        return [grad.clamp(min=0) / grad.max() for grad in self.gradients]

    def negative_saliency(self):
        """Returns negative gradients"""
        return [(-grad).clamp(min=0) / - grad.min() for grad in self.gradients]

    def grad_x_input(self):
        """Returns the gradient multiplied by the input image"""
        return [img * grad for img, grad in zip(self.input, self.gradients)]

    def __call__(self, images, classes=None):
        """

        Parameters
        ----------
        images: List
            list of images used to generate saliency map

        classes: Optional[List]
            class index of each images, if not provided will default to class = 0.
            `ImageNet class to index map <https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a>`_
        """
        self.input = self.preprocessor(images)
        self.input.requires_grad = True

        out = self.model(self.input)
        self.model.zero_grad()

        # if no classes are provided use a dummy one
        if classes is None:
            classes = [0 for _ in range(len(images))]

        # Fabricate gradient to back-propagate
        gradient = torch.zeros_like(out, dtype=torch.float32)
        for i, cls in enumerate(classes):
            gradient[i, cls] = 1

        out.backward(gradient=gradient)
        return out
