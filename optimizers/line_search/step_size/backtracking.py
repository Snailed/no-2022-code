import numpy as np


def backtrack(f, x, p, f_grad, a_bar=1, rho=0.9, c=0.2):
    a = a_bar
    iterations = 0
    while f(x + a*p) > f(x) + c * a * (np.array(f_grad(x)).T @ p):
        a = rho * a
        iterations += 1
        assert iterations < 100000
    return a, iterations


def test():
    def f(x):
        return np.sum([x_i**2 for x_i in x])

    def f_grad(x):
        return np.array([2*x_i for x_i in x])

    init_x = [0.5, 0.5]
    # the optimal step size is less than 1
    assert backtrack(f, init_x, np.array(
        [-1, -1]), f_grad, a_bar=1, rho=0.7, c=0.5)[0] < 1

    assert backtrack(f, [1, 1], np.array(
        [-1.9, -1.9]), f_grad, a_bar=1, rho=0.9, c=0.5)[0] < 1
