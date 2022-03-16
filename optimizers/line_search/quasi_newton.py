import numpy as np
from scipy.optimize import line_search

def backtrack(f, x, p, f_grad, a_bar=10, rho=0.9, c=0.5):
    a = a_bar
    iterations = 0
    while f(x + a*p) > f(x) + c * a * (np.array(f_grad(x)).T @ p):
        a = rho * a
        iterations += 1
        assert iterations < 100000
    return a, iterations

class QuasiNewtonLineSearch:
    @staticmethod
    def get_label():
        return 'Quasi-Newton Line Search'

    @staticmethod
    def minimize(
            f,
            x0,
            der,
            max_iterations=10000,
            callback=None,
            initial_alpha=0.1,
            initial_inv_hessian=None,
            c1=1e-4, c2=0.9,
    ):

        x = np.array(x0)
        alpha = initial_alpha
        inv_hessian = initial_inv_hessian
        if inv_hessian is None:
            inv_hessian = np.linalg.inv(np.eye(len(x0)))

        for k in range(max_iterations):
            p_k = -inv_hessian @ der(x)
            p_k_scale_factor, _ = backtrack(f, x, p_k, der)
            p_k = p_k @ p_k_scale_factor
            alpha, _, _, new_f_val, old_f_val, new_slope = line_search(f, der, x, p_k, c1=c1, c2=c2)
            new_x = x + alpha * p_k
            y_k = np.array(new_slope) - np.array(der(x))
            s_k = new_x - x
            rho_k = 1/(y_k.T @ s_k)
            I = np.eye(len(x))
            inv_hessian = (I - rho_k * s_k @ y_k.T) @ inv_hessian @ (I -
                                                             rho_k * y_k @ s_k.T) + rho_k * s_k @ s_k.T
            x = new_x
            if callback and callback(x):
                return x
