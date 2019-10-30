from olympus.utils.factory import fetch_factories


factories = fetch_factories('olympus.optimizers', __file__)


def get_optimizer_builder(optimizer_name):
    return factories[optimizer_name]()
