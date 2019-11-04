import torch.nn as nn

from olympus.utils import MissingArgument
from olympus.utils.factory import fetch_factories
from olympus.utils.fp16 import network_to_half

registered_models = fetch_factories('olympus.models', __file__)


def known_models():
    return registered_models.keys()


class RegisteredModelNotFound(Exception):
    pass


class Model(nn.Module):
    """Olympus standardized Model interface

    Parameters
    ----------
    name: str
        Name of a registered model

    half: bool
        Convert the network to half/fp16

    model: Model
        Custom model to use, mutually exclusive with :param name

    Raises
    ------
    RegisteredModelNotFound
        when using a name of an known model

    MissingArgument:
        if name nor model were not set
    """

    def __init__(self, name=None, half=False, model=None, input_size=None, output_size=None):
        super(Model, self).__init__()

        # Override the model with a custom model
        if model:
            self.model = model

        elif name:
            # load an olympus model
            self.model = registered_models.get(name)(input_size=input_size, output_size=output_size)

            if not self.model:
                raise RegisteredModelNotFound(name)

        else:
            raise MissingArgument('Model or Name needs to be set')

        self.transform = lambda x: x
        self.half = half

        if half:
            self.model = network_to_half(model)
            self.transform = lambda x: x.half()

    def forward(self, input):
        return self.model(self.transform(input))

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        destination = {
            'model': self.model.state_dict(None, prefix, keep_vars),
            'half': self.half
        }
        return destination

    def load_state_dict(self, state_dict, strict=True):
        self.half = state_dict['half']
        if self.half:
            self.transform = lambda x: x.half()
        self.model.load_state_dict(state_dict['model'], strict=strict)

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse)

