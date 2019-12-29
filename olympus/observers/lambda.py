from olympus.observers.observer import Observer


def _wrapper_maker(function):
    """Ignore self"""
    def clean(self, *args, **kwargs):
        return function(*args, **kwargs)
    return clean


class LambdaObserver(Observer):
    def __init__(self, event_name, function):
        self.event_name = event_name
        self.function = function

        setattr(self, f'on_{event_name}', _wrapper_maker(self.function))




