import numpy as np
from scipy.optimize import minimize


class ScipyBFGS:
    @staticmethod
    def get_label():
        return 'SciPy BFGS'

    @staticmethod
    def minimize(f, x0, der, callback=None):
        optimization_result = minimize(f, x0, method='BFGS', jac=der, tol=None, callback=callback, options={
            'gtol': 1e-2, 'disp': False})
        if optimization_result.success:
            return optimization_result.x, optimization_result.nit, optimization_result.jac
        else:
            if optimization_result.message == 'Desired error not necessarily achieved due to precision loss.':
                return optimization_result.x, optimization_result.nit, optimization_result.jac
            else:
                print('Optimization failure message:')
                print(optimization_result.message)
                print('Optimization point:')
                print(optimization_result.x)
                print('Optimization starting point:')
                print(x0)
                raise AssertionError('Optimization failed!')
