import numpy as np
from functions.ellipsoid import Ellipsoid
from utils.validate_gradient import assert_gradient


class LogEllipsoid:
    @staticmethod
    def f(x, epsilon=0.0001, alpha=1000):
        return np.log(epsilon + Ellipsoid.f(x, alpha))

    @staticmethod
    def gradient(x, epsilon=0.0001, alpha=1000):
        return Ellipsoid.gradient(x, alpha)/(epsilon + Ellipsoid.f(x, alpha))


def test():
    assert LogEllipsoid.f([0, 1, 2]) == 8.301924272787891
    assert LogEllipsoid.f([6, 10, 100]) == 16.118415427600336
    assert LogEllipsoid.f([-8, 7], epsilon=0.1,
                          alpha=700) == 10.444767693775828

    assert_gradient(
        lambda x: LogEllipsoid.gradient(x),
        lambda x: LogEllipsoid.f(x),
        [np.linspace(-30+i*10, 30+i*10, 100) for i in range(0, 40)], 0.00001, threshold=0.01)
