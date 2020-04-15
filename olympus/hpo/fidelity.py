from dataclasses import dataclass


@dataclass
class Fidelity:
    """Fidelity, a metric that is representative of the level of training a trial has done.

    Parameters
    ----------
    min: int
        minimum value of the fidelity

    max: int
        maximum value of the fidelity (maximum epoch)

    base: int
        Base logarithm of the fidelity dimension.

    Notes
    -----

    How this parameter is used is defined by the HP optimizers themselves.
    For example random search only use ``fidelity.max``.
    The examples below only show how it could be used.

    Examples
    --------

    Smaller base creates more stages `fidelity('epoch', 1, 100, base=2)`

    .. code-block:: text
        # `fidelity('epoch', 1, 100, base=2)`

           epoch  trial_count
        0      2          448
        1      4          224
        2      7          112
        3     13           56
        4     25           28
        5     50           14
        6    100            7
        Total: 889

    .. code-block:: text
        # `fidelity('epoch', 1, 100, base=3)`

           epoch  trial_count
        0      2          405
        1      4          135
        2     12           45
        3     34           15
        4    100            5
        Total:  605

    """
    min: int
    max: int
    base: int = 2
    name: str = 'epoch'

    def __repr__(self):
        return f'fidelity({self.min}, {self.max}, {self.base})'

    def to_dict(self):
        return {
            'min': self.min,
            'max': self.max,
            'base': self.base,
            'name': self.name,
        }

    @staticmethod
    def from_dict(data):
        return Fidelity(**data)


def fidelity(name, min, max, base):
    return Fidelity(min, max, base, name)
