import numpy as np
from optimizers.line_search.direction.direction_matrix import steepest_gradient
from optimizers.line_search.step_size import backtracking
from optimizers.line_search import line_search


def backtrack(*args):
    return backtracking.backtrack(*args)[0]


class SteepestDescentLineSearch:
    @staticmethod
    def get_label():
        return 'Steepest Descent Line Search'

    @staticmethod
    def minimize(
            f,
            x0,
            der,
            hes=None,
            max_iterations=10000,
            callback=None,
            step_size_f=backtrack
    ):
        def direction_matrix_f(*_):
            return steepest_gradient(len(x0))

        def default_callback(xk):
            if callback:
                return callback(xk)
            else:
                if np.linalg.norm(der(xk)) < 0.00001:
                    return True
                return False

        return line_search(
            f,
            x0,
            der,
            direction_matrix_f,
            step_size_f=step_size_f,
            max_iterations=max_iterations,
            callback=default_callback,
            f_hess=None
        )


def test():
    def f(x):
        return np.sum([x_i ** 2 for x_i in x])

    def f_grad(x):
        return np.array([2 * x_i for x_i in x])

    assert np.abs(SteepestDescentLineSearch.minimize(
        f, [2], f_grad)[0]) < 0.00001
    assert np.abs(SteepestDescentLineSearch.minimize(
        f, [2, 2], f_grad)[1]) < 0.00001
