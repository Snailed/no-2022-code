import numpy as np
from optimizers.line_search.direction.direction_matrix import newton
from optimizers.line_search.step_size import backtracking
from optimizers.line_search import line_search


def backtrack(*args):
    return backtracking.backtrack(*args)[0]


class NewtonLineSearch:
    @staticmethod
    def get_label():
        return 'Newton Line Search'

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
        assert hes is not None

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
            newton,
            step_size_f=step_size_f,
            max_iterations=max_iterations,
            callback=default_callback,
            f_hess=hes
        )


def test():
    def f(x):
        return np.sum([x_i ** 2 for x_i in x])

    def f_grad(x):
        return np.array([2 * x_i for x_i in x])

    def f_hes(x):
        return np.diag([2 for _ in x])

    assert np.abs(NewtonLineSearch.minimize(
        f, [2], f_grad, hes=f_hes)[0]) < 0.00001
    assert np.abs(NewtonLineSearch.minimize(
        f, [2, 2], f_grad, hes=f_hes)[1]) < 0.00001
