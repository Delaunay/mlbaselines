from olympus.utils.factory import fetch_factories
from olympus.utils.fp16 import ModelAdapter

factories = fetch_factories('olympus.models', __file__)


def build_model(name=None, half=False, **kwargs):
    return ModelAdapter(
        factories[name](**kwargs),
        half=half
    )
