import numpy as np
from utils.validate_gradient import assert_gradient

class SumOfDifferentPowers:
    @staticmethod
    def f(x):
        return np.sum([(x_i**2)**(1+(((i + 1) - 1)/(len(x)-1))) for i, x_i in enumerate(x)])

    @staticmethod
    def gradient(x):
        d = len(x)

        def bottum(d, i):
            return ((2*d + 2*(i+1) - 4)*(d + 2*(i+1) - 3))

        def exponent(d, i):
            return ((2*(i+1) - 2)/(d - 1))

        def potens(x_i, d, i):
            return x_i**exponent(d, i)
        return [(bottum(d, i)/(d-1)**2) * potens(x_i, d, i) for i, x_i in enumerate(x)]


def test():
    assert SumOfDifferentPowers.f([1,2]) == 1**2**1 + 2**2**2
    assert SumOfDifferentPowers.f([1,2,3]) == 1**2 + (2**2)**(1+1/2) + 3**2**2
    assert SumOfDifferentPowers.f([5,4,3,2]) == 5**2 + (4**2)**(1+1/3) + (3**2)**(1+2/3) + (2**2)**2

    # Note that this only tests a positive domain - a negative domain on the gradient might result in evaluation of complex numbers (for example, (-1)**(1/2) )
    assert_gradient(
        lambda x: SumOfDifferentPowers.gradient(x),
        lambda x: SumOfDifferentPowers.f(x),
        [np.linspace(0, 1, 100)], 0.000001, threshold=1)

    assert_gradient(
        lambda x: SumOfDifferentPowers.gradient(x),
        lambda x: SumOfDifferentPowers.f(x),
        [np.linspace(1, 2, 100)], 0.000001, threshold=0.5)

    assert_gradient(
        lambda x: SumOfDifferentPowers.gradient(x),
        lambda x: SumOfDifferentPowers.f(x),
        [np.linspace(2, 4, 100)], 0.0000001, threshold=0.5)
