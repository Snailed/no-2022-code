from optimizers.line_search.direction import direction_from_inv_matrix


def line_search(f, x_init, f_grad, direction_matrix_f, step_size_f, max_iterations=10000, callback=None, f_hess=None):
    x = x_init
    iterations = 0
    for k in range(max_iterations):
        p = direction_from_inv_matrix(f_grad, x, direction_matrix_f(f_hess, x))
        alpha = step_size_f(f, x, p, f_grad)
        # print("x", x, "p", p, "alpha", alpha, "x + alpha * p", x + alpha * p)
        x = x + alpha * p
        if callback is not None and callback(x):
            break
        iterations = k
    return x, iterations, f_grad(x)
