import numpy as np
from optimizers.line_search.direction.direction_matrix import steepest_gradient, newton


def direction_from_inv_matrix(f_grad, x, inv_matrix):
    return -1 * (inv_matrix @ f_grad(x))


def test():
    def f(x):
        return np.sum([x_i**2 for x_i in x])

    def f_grad(x):
        return np.array([2 * x_i for x_i in x])

    def f_hess(x):
        return np.diag([2 for _ in x])

    assert direction_from_inv_matrix(f_grad, [1, 1], steepest_gradient(
        2))[0] == -2
    assert direction_from_inv_matrix(f_grad, [1, 1], steepest_gradient(
        2))[1] == -2

    # make sure it chooses the right direction
    assert direction_from_inv_matrix(
        f_grad, [1, 1], newton(f_hess, [1, 1]))[0] < 0
    assert direction_from_inv_matrix(
        f_grad, [1, 1], newton(f_hess, [1, 1]))[1] < 0
