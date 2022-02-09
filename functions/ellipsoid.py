import numpy as np
from utils.validate_gradient import assert_gradient
# from numpy.testing import assert_array_equal


class Ellipsoid:
    @staticmethod
    def f(x, alpha=1000):
        def sum_func(x_i, i):
            return pow(alpha, (i - 1)/(len(x) - 1))*pow(x_i, 2)
        sum_parts = [sum_func(x_i, i + 1) for i, x_i in enumerate(list(x))]
        return np.sum(list(sum_parts))

    @staticmethod
    def gradient(x, alpha=1000):
        return [
                (2*alpha**(((i + 1)-1)/(len(x) - 1))*x_i)
                for i, x_i in enumerate(x)
            ]


def test():
    assert Ellipsoid.f([0, 1, 2]) == 0 + pow(1000, 1/2) + pow(1000, 2/2) * 4
    assert Ellipsoid.f([6, 10, 100]) == pow(
        6, 2) + pow(1000, 1/2)*pow(10, 2) + pow(1000, 2/2) * pow(100, 2)
    assert Ellipsoid.f([-8, 7], alpha=700) == pow(-8, 2) + \
        pow(700, 1) * pow(7, 2)
    assert_gradient(
        lambda x: Ellipsoid.gradient(x),
        lambda x: Ellipsoid.f(x),
        [np.linspace(-30+i*10, 30+i*10, 100) for i in range(0, 40)], 0.00001, threshold=0.001)
