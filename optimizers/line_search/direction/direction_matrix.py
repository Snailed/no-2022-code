import numpy as np


def steepest_gradient(N):
    return np.eye(N)


def newton(f_hess, x):
    eigenvals = np.linalg.eigvals(f_hess(x))
    taus = [1 - l if l < 1 else 0 for l in eigenvals]
    # print('newton eigenvals', eigenvals)
    if np.all(eigenvals > 0):
        # print('INVERTABLE!')
        return np.linalg.inv(f_hess(x))
    else:
        # print('NON-INVERTABLE!')
        inner_new_matrix = f_hess(x) + taus*np.eye(len(x))
        new_matrix = np.linalg.inv(inner_new_matrix)
        # print('new matrix', new_matrix)
        # print('Is the new matrix psd?', np.all(
        # np.linalg.eigvals(new_matrix) > 0))
        return new_matrix
