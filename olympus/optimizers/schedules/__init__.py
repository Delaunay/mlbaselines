from olympus.utils.factory import fetch_factories


factories = fetch_factories('olympus.optimizers.schedules', __file__)


def get_schedule_builder(schedule_name):
    return factories[schedule_name]()
