

class ExponentialSmoothing:
    """More on `Wikipedia <https://en.wikipedia.org/wiki/Exponential_smoothing>`_"""

    def __init__(self, alpha, value=None):
        assert 0 < alpha < 1, 'alpha must be between 0 and 1'

        self.alpha = alpha
        self.value = value

    def __iadd__(self, other):
        if self.value is None:
            self.value = other
            return self

        self.value = self.alpha * other + (1 - self.alpha) * self.value
        return self

    def __str__(self):
        return f'ESmooth<{self.value}>'


class MovingAverage:
    """More on `Wikipedia MA <https://en.wikipedia.org/wiki/Moving_average>`_"""

    def __init__(self, n: int):
        self.n = n
        self.values = [None] * n
        self.pos = 0

    def __iadd__(self, other):
        self.values[self.pos % self.n] = other
        self.pos += 1
        return self

    @property
    def value(self):
        if self.pos < self.n:
            return sum(self.values[:self.pos]) / self.pos

        return sum(self.values) / self.n

    def __str__(self):
        return f'MA<{self.value}>'
