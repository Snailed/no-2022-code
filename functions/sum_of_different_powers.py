import numpy as np
from utils.validate_gradient import assert_gradient


class SumOfDifferentPowers:
    @staticmethod
    def f(x):
        return np.sum([(x_i**2)**(1+(((i + 1) - 1)/(len(x)-1))) for i, x_i in enumerate(x)])

    @staticmethod
    def gradient(x):
        d = len(x)
        return [(1 + ((i + 1)-1)/(d-1))*(x_i**2)**(((i + 1)-1)/(d-1))*2*x_i for i, x_i in enumerate(x)]

    @staticmethod
    def hessian(x):
        d = len(x)
        matrix = np.zeros((len(x), len(x)))
        for i in range(1, len(x)+1):
            matrix[i-1][i-1] = (((2*d + 2*i - 4)*(d + 2*i - 3)) /
                                ((d-1)**2))*x[i-1]**((2*i-2)/(d-1))
        return matrix


def test():
    assert SumOfDifferentPowers.f([1, 2]) == 1**2**1 + 2**2**2
    assert SumOfDifferentPowers.f(
        [1, 2, 3]) == 1**2 + (2**2)**(1+1/2) + 3**2**2
    assert SumOfDifferentPowers.f(
        [5, 4, 3, 2]) == 5**2 + (4**2)**(1+1/3) + (3**2)**(1+2/3) + (2**2)**2

    # Note that this only tests a non-negative domain - a negative domain on the gradient might result in evaluation of complex numbers (for example, (-1)**(1/2) )
    assert_gradient(
        lambda x: SumOfDifferentPowers.gradient(x),
        lambda x: SumOfDifferentPowers.f(x),
        [np.linspace(30*i, 30*i + 30, 100) for i in range(0, 30)], 0.000001, threshold=0.5)
