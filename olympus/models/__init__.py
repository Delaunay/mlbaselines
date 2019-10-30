from olympus.utils.factory import fetch_factories


factories = fetch_factories('olympus.models', __file__)


def build_model(name=None, **kwargs):
    return factories[name](**kwargs)
