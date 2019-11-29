from olympus.accumulators.smoothing import ExponentialSmoothing, MovingAverage


def test_smoothing():
    smooth = ExponentialSmoothing(alpha=0.5)

    for i in range(0, 10):
        smooth += i

    assert smooth.value == 8.001953125


def test_moving_average():
    ma = MovingAverage(10)

    for i in range(0, 20):
        ma += i

    assert ma.value == 14.5
