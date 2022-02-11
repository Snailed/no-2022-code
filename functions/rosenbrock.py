import numpy as np
from utils.validate_gradient import assert_gradient


class Rosenbrock:
    @staticmethod
    def f(x: [float, float]):
        return pow(1 - x[0], 2) + 100 * pow(x[1] - pow(x[0], 2), 2)

    @staticmethod
    def gradient(x: [float, float]):
        return np.array([
            2*x[0] - 2 + 400*x[0]**3 - 400*x[0]*x[1],
            200*x[1] - 200*x[0]**2
        ])

    @staticmethod
    def hessian(x):
        return np.array(
            [
                [2 + 1200*x[0]**2 - 400*x[1]**4, -400*x[0]],
                [-400*x[0], 200]
            ]
        )

def test():
    assert Rosenbrock.f([1, 1]) == 0.0
    assert Rosenbrock.f([200, 500]) == 156025039601.0
    assert Rosenbrock.f([-1.242, 572.356]) == 32582802.89837661
    assert_gradient(
        lambda x: Rosenbrock.gradient(x),
        lambda x: Rosenbrock.f(x),
        [[0, 1], [-1, 3], [0.001, 0.004], [-30, 23]],
        0.000000001)
