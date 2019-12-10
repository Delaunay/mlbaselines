# -*- coding: utf-8 -*-
"""
:mod:`olympus.resuming -- Resuming Aspect
=========================================
.. module:: resuming
   :platform: Unix
   :synopsis: Implement the Interface and Entry point for saving states and resuming from states
"""
from olympus.utils import warning, debug
from olympus.utils.functional import select


class Aspect:
    pass


class Resumable(Aspect):
    _aspects = {}

    def load_state_dict(self, obj, storage, strict):
        raise NotImplementedError()

    def state_dict(self, obj, destination=None, prefix='', keep_vars=False):
        raise NotImplementedError()

    @staticmethod
    def add_apsect(obj_type, aspect):
        if obj_type in aspect:
            warning(f'Overriding the aspect of {obj_type}')

        Resumable._aspects[obj_type] = aspect

    @staticmethod
    def get_aspect(obj_type, default):
        return Resumable._aspects.get(obj_type, default)


class DefaultResumingAspect(Resumable):
    """Iterates over an object attributes and look for resumable attributes"""

    def load_state_dict(self, obj, states, strict):
        missing_fields = []

        for field_name, field in obj.__dict__.items():
            if hasattr(field, 'load_state_dict'):
                if field_name in states:
                    load_state_dict(field, states[field_name], strict)
                else:
                    missing_fields.append(field_name)

        if strict and missing_fields:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                obj.__class__.__name__, "\n\t".join(missing_fields)))

    def state_dict(self, obj, destination=None, prefix='', keep_vars=False):
        state = select(destination, {})

        for field_name, field in obj.__dict__.items():
            if hasattr(field, 'state_dict'):
                try:
                    state[field_name] = state_dict(field, destination, prefix, keep_vars)
                except:
                    print(f'An error occurred when trying {field_name}')
                    raise

        return state


class BadResume(Exception):
    pass


class BadResumeGuard:
    """Make sure we do not reuse a task with a bad state"""

    def __init__(self, task):
        self.task = task

    def __enter__(self):
        if hasattr(self.task, 'bad_state') and self.task.bad_state:
            raise BadResume('Cannot resume from bad state! '
                            'You need to create a new task than can resume the previous state')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and hasattr(self.task, 'bad_state'):
            self.task.bad_state = True


def load_state_dict(obj, state_dict, strict=True, force_default=False):
    with BadResumeGuard(obj):
        if hasattr(obj, 'load_state_dict') and not force_default:
            return obj.load_state_dict(state_dict, strict)

        aspect = Resumable.get_aspect(type(obj), DefaultResumingAspect())
        return aspect.load_state_dict(obj, state_dict, strict)


def state_dict(obj, destination=None, prefix='', keep_vars=False, force_default=False):
    with BadResumeGuard(obj):
        # class override
        if hasattr(obj, 'state_dict') and not force_default:
            return obj.state_dict(destination, prefix, keep_vars)

        aspect = Resumable.get_aspect(type(obj), DefaultResumingAspect())
        return aspect.state_dict(obj, destination, prefix, keep_vars)

