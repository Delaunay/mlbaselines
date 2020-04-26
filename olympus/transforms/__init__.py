import torch
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms


def minimize(size):
    return transforms.Compose([
        to_pil_image,
        transforms.Resize(7),
        transforms.ToTensor()])


class Preprocessor:
    """List of function called on the network inputs before"""
    def __init__(self, *preprocessors):
        self.preprocessors = list(preprocessors)

    def append(self, preprocessor):
        self.preprocessors.append(preprocessor)

    def forward(self, args):
        output = args

        for pre in self.preprocessors:
            output = pre(output)

        return output

    def __call__(self, args):
        return self.forward(args)

    def parameters(self):
        """Returns a list of parameters"""
        param_list = []
        for preprocessor in self.preprocessors:
            if hasattr(preprocessor, 'parameters'):
                param_list.append({
                    'params': list(preprocessor.parameters())
                })

        return param_list

    def get_space(self):
        return {}

    def init(self, **kwargs):
        pass

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_list = []
        for pre in self.preprocessors:
            if hasattr(pre, 'state_dict'):
                state_list.append(pre.state_dict(destination, prefix, keep_vars))
            else:
                state_list.append({})

        return state_list

    def load_state_dict(self, states, strict=True):
        for pre, state in zip(self.preprocessors, states):
            if hasattr(pre, 'load_state_dict'):
                pre.load_state_dict(state)


class ConcatPreprocessor(Preprocessor):
    """Concatenate multiple preprocessors into a single one"""
    def __init__(self, *preprocessors):
        super(ConcatPreprocessor, self).__init__()

        for pre in preprocessors:
            self.preprocessors.extend(pre.preprocessors)


class _PreprocessingNode:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, arg):
        raise NotImplementedError()

    def parameters(self):
        return self.transform.parameters()

    def get_space(self):
        return self.transform.get_space()

    def init(self, **kwargs):
        return self.transform.init(**kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.transform.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, states, strict=True):
        return self.transform.load_state_dict(states)


class DimSelect(_PreprocessingNode):
    """Execute a transformation to a specified dimension only

    Examples
    --------
    >>> shape = 3, 20, 20
    >>>
    >>> p = Preprocessor(
    >>>    BatchedTransform(ToPILImage(), dim=0),
    >>>    BatchedTransform(ToTensor(), dim=0),
    >>>    DimSelect(SpatialTransformerNetwork(input_shape=shape), dim=0))
    >>>
    >>> batch_size = 256
    >>> image = torch.randn((batch_size,) + shape)
    >>> target = torch.randn((batch_size,) + (1,))
    >>>
    >>> input = (image, target)
    >>>
    >>> out = p(input)  # only image is preprocessed
    """
    def __init__(self, transform, dim=-1):
        super(DimSelect, self).__init__(transform)
        self.dim = dim

    def __call__(self, x):
        if self.dim > -1 and hasattr(x, '__len__') and len(x) > 1:
            return tuple(self.transform(val) if i == self.dim else val for i, val in enumerate(x))

        return self.transform(x)


class BatchedTransform(_PreprocessingNode):
    """Execute individual transformation to batch of samples

    Examples
    --------
    >>> shape = 3, 20, 20
    >>>
    >>> p = Preprocessor(BatchedTransform(ToPILImage(), dim=0))
    >>>
    >>> batch_size = 256
    >>> image = torch.randn((batch_size,) + shape)
    >>> target = torch.randn((batch_size,) + (1,))
    >>>
    >>> input = (image, target)
    >>>
    >>> out = p(input)  # only the image tensor is passed to ToPILImage
    """
    def __init__(self, transform, dim=-1):
        super(BatchedTransform, self).__init__(transform)
        self.dim = dim

    def __call__(self, x):
        if self.dim > -1 and hasattr(x, '__len__'):
            return tuple(self.all(val) if i == self.dim else val for i, val in enumerate(x))

        return self.all(x)

    def all(self, x):
        samples = []
        for sample in x:
            samples.append(self.transform(sample))

        # if samples are tensors concatenate them to for a single tensor
        if isinstance(samples[0], torch.Tensor):
            return torch.stack(samples)

        return samples


class Denormalize:
    def __init__(self, mean, std, inplace=False):
        self.mean = - torch.as_tensor(mean, dtype=torch.float32)
        self.std = 1.0 / torch.as_tensor(std, dtype=torch.float32)
        self.inplace = inplace

    def __call__(self, tensor):
        if not self.inplace:
            tensor = tensor.clone()

        tensor.div_(self.std[:, None, None])
        tensor.sub_(self.mean[:, None, None])
        return tensor

