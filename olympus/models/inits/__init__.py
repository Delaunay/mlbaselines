from olympus.utils.factory import fetch_factories

factories = fetch_factories('olympus.models.inits', __file__)


def make_init(model, name=None, half=False, **kwargs):
    print(name, factories[name])
    return factories[name](model, **kwargs)
