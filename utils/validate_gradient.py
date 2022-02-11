import numpy as np
from utils.error_measurements import relative_error

def slope(f, x, h):
    # assuming h is a n size vector containing all 0's except for one coordinate
    return abs(f(x + h) - f(x))/np.linalg.norm(h)

def assert_gradient(expected_nabla_f, f, xs, h, threshold=0.0001):
    # Threshold is arbitrarily chosen to be some small value
    for x in xs:
        for i, _ in enumerate(x):
            # h_vec is a vector that makes sure a small step is taken in direction x_i
            h_vec = np.zeros(len(x))
            h_vec[i] = h
            if relative_error(abs(expected_nabla_f(x)[i]), slope(f, x, h_vec)) > threshold:
                print(expected_nabla_f(x)[i], slope(f, x, h_vec))
                print(relative_error(abs(expected_nabla_f(x)[i]), slope(f, x, h_vec)), threshold)
            assert relative_error(abs(expected_nabla_f(x)[i]), slope(f, x, h_vec)) < threshold

def assert_hessian(expected_nabla_nabla_f, nabla_f, xs, h, threshold=0.0001):
    for x in xs:
        for i in range(1, len(x) + 1):
            for j in range(1, len(x) + 1):
                h_vec = np.zeros(len(x))
                h_vec[j] = h
                if relative_error(abs(expected_nabla_nabla_f(x)[i][j]), slope(lambda x: nabla_f(x)[j], x, h_vec)) > threshold:
                    print(expected_nabla_nabla_f(x)[i][j], slope(lambda x: nabla_f(x)[j], x, h_vec))
                    print(relative_error(abs(expected_nabla_nabla_f(x)[i][j]), slope(lambda x: nabla_f(x)[j], x, h_vec)))
                assert relative_error(abs(expected_nabla_nabla_f(x)[i][j]), slope(lambda x: nabla_f(x)[j], x, h_vec)) < threshold


def test():
    def f(x):
        return np.sum([x_i**2 for x_i in x])

    def gradient(x):
        return [2*x_i for x_i in x]

    try:
        assert_gradient(gradient, f, [[1, 2, 3, 4], [2, 3, 4, 5], [5, 6, 7, 8]], 0.0001)
        assert False  # step size should cause assertion fail
    except AssertionError:
        pass
    assert_gradient(gradient, f, [[1, 2, 3, 4], [2, 3, 4, 5], [5, 6, 7, 8]], 0.00001)
