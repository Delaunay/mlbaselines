import copy

import numpy


def select(a, b):
    if a is not None:
        return a
    return b


def flatten(dictionary):
    """Turn all nested dict keys into a {key}.{subkey} format"""
    def _flatten(dictionary):
        if dictionary == {}:
            return dictionary

        key, value = dictionary.popitem()
        if not isinstance(value, dict) or not value:
            new_dictionary = {key: value}
            new_dictionary.update(_flatten(dictionary))
            return new_dictionary

        flat_sub_dictionary = _flatten(value)
        for flat_sub_key in list(flat_sub_dictionary.keys()):
            flat_key = key + '.' + flat_sub_key
            flat_sub_dictionary[flat_key] = flat_sub_dictionary.pop(flat_sub_key)

        new_dictionary = flat_sub_dictionary
        new_dictionary.update(_flatten(dictionary))
        return new_dictionary

    return _flatten(copy.deepcopy(dictionary))


def unflatten(dictionary):
    """Turn all keys with format {key}.{subkey} into nested dictionaries"""
    unflattened_dictionary = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        sub_dictionary = unflattened_dictionary
        for part in parts[:-1]:
            if part not in sub_dictionary:
                sub_dictionary[part] = dict()
            sub_dictionary = sub_dictionary[part]
        sub_dictionary[parts[-1]] = value
    return unflattened_dictionary


def encode_rng_state(state):
    state = list(state)
    state[1] = state[1].tolist()
    return tuple(state)


def decode_rng_state(state):
    state = list(state)
    state[1] = numpy.array(state[1])
    return tuple(state)
