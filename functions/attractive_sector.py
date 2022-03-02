import numpy as np
from utils.validate_gradient import assert_gradient


class AttractiveSector:
    @staticmethod
    def _h(x, q):
        return (np.log(1 + np.exp(-np.abs(q * x))) + max(q*x, 0))/q

    @staticmethod
    def _h_prime(x, q):
        return (np.exp(-np.abs(x*q)) + max(x*q, 0))/(1 + np.exp(-np.abs(x*q)) + max(x*q, 0))

    @staticmethod
    def _h_prime_prime(x, q):
        return ((np.exp(-np.abs(x*q)) + max(x*q, 0))*q)/((1 + np.exp(-np.abs(x*q)) + max(x*q, 0))**2)

    @staticmethod
    def f(x, q=10000):
        array = [AttractiveSector._h(
            x_i, q)**2 + 100*AttractiveSector._h(-x_i, q)**2 for x_i in x]
        return np.sum(array)

    @staticmethod
    def gradient(x, q=10000):
        return [
            2*AttractiveSector._h(x_i, q) * AttractiveSector._h_prime(x_i, q)
            + 200*AttractiveSector._h(-x_i, q) *
            AttractiveSector._h_prime(-x_i, q)
            for x_i in x
        ]

    @staticmethod
    def hessian(x, q=10000):
        matrix = np.zeros((len(x), len(x)))
        def h(x, q): return AttractiveSector._h(x, q)
        def h_p(x, q): return AttractiveSector._h_prime(x, q)
        def h_p2(x, q): return AttractiveSector._h_prime_prime(x, q)
        for i in range(0, len(x)):
            matrix[i][i] = 2*(h_p(x[i], q)**2) + 2*h(x[i], q)*h_p2(x[i], q) + \
                200 * (h_p(-x[i], q)**2) + 200 * h(-x[i], q)*h_p2(-x[i], q)
        return matrix


def test():
    # Test exp hack
    def exp_before(x):
        return np.log(1 + np.exp(x))

    def exp_after(x):
        return np.log(1 + np.exp(-np.abs(x))) + max(x, 0)

    assert exp_before(23) == exp_after(23)

    assert AttractiveSector._h(1, 10000) == 1.0
    assert AttractiveSector._h(0.0003, 10000) == 0.0003048587351573742
    assert AttractiveSector._h(0.00003, 10000) == 8.54355244468527e-05

    assert AttractiveSector._h(7, 50) == np.log(1 + np.exp(50 * 7))/50
    # TODO: Test of numerical stability!

    assert AttractiveSector.f([0.0000001, 0.2], q=1023) == 0.04004636157704798
    # Test around 0
    assert AttractiveSector.f(
        [-1, -0.1, -0.00001, 0.0, 0.00001, 0.1, 1]) == 102.01000146432473

    assert_gradient(
        lambda x: AttractiveSector.gradient(x),
        lambda x: AttractiveSector.f(x),
        [np.linspace(-30+i*10, 30+i*10, 100) for i in range(0, 10)], 0.00001, threshold=0.1)
